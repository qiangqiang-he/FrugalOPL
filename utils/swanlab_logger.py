"""Controller-owned SwanLab logging support for FrugalOPL.

One FrugalOPL training run maps to exactly one SwanLab experiment. Only the
Controller owns this logger and writes to SwanLab; it collects telemetry from
Workers through the framework's normal messages. Every Worker metric uses its
configured pipeline node ID as its namespace, such as ``teachera/`` or
``teacherb/``. Algorithm metrics use the experiment's ``algorithm.algo_name``
namespace instead, such as ``pg_opd/``; evaluation remains a framework-level
namespace because it is not a pipeline Worker.

The module intentionally never accepts an API key through configuration. An
enabled run requires ``SWANLAB_API_KEY`` to already be present in the
Controller process environment before the training launcher starts.
"""

from __future__ import annotations

import copy
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from engine.pipeline import TrainingPipeline


SwanLabState = Literal["success", "crashed", "aborted"]
_FRAMEWORK_NAMESPACES: frozenset[str] = frozenset({"algorithm", "evaluation"})


def _mapping(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    return dict(value)


def _optional_text(value: object, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string when configured.")
    return value.strip()


def _pipeline_node_namespaces(pipeline: "TrainingPipeline") -> frozenset[str]:
    """Return the exact Worker namespaces declared by the resolved pipeline."""

    namespaces = frozenset(node.node_id for node in pipeline.nodes)
    if not namespaces or any("/" in namespace for namespace in namespaces):
        raise ValueError("Pipeline node IDs used for SwanLab metrics must be non-empty and contain no '/'.")
    if namespaces & _FRAMEWORK_NAMESPACES:
        raise ValueError("Pipeline node IDs must not conflict with SwanLab framework namespaces.")
    return namespaces


def _algorithm_namespace(config: Mapping[str, object], *, reserved_namespaces: frozenset[str]) -> str:
    """Read the per-experiment SwanLab namespace for algorithm metrics."""

    algorithm_cfg = _mapping(config.get("algorithm"), "algorithm")
    algo_name = _optional_text(algorithm_cfg.get("algo_name"), "algorithm.algo_name")
    if algo_name is None:
        raise ValueError("algorithm.algo_name must be a non-empty string.")
    if "/" in algo_name:
        raise ValueError("algorithm.algo_name must not contain '/'.")
    if algo_name in reserved_namespaces:
        raise ValueError(
            "algorithm.algo_name must not reuse a framework SwanLab namespace: "
            f"{algo_name!r}."
        )
    return algo_name


def _scalar_number(value: object) -> float | int | None:
    """Return a finite scalar without importing Torch or NumPy eagerly."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None

    item = getattr(value, "item", None)
    numel = getattr(value, "numel", None)
    if callable(item):
        if callable(numel):
            try:
                if int(numel()) != 1:
                    return None
            except Exception:
                return None
        try:
            return _scalar_number(item())
        except Exception:
            return None
    return None


def _flatten_numeric_metrics(metrics: Mapping[str, object], *, prefix: str = "") -> dict[str, float | int]:
    """Flatten nested metric mappings and discard non-scalar values safely."""

    flattened: dict[str, float | int] = {}
    for raw_key, value in metrics.items():
        key = str(raw_key).strip().strip("/")
        if not key:
            continue
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(value, Mapping):
            flattened.update(_flatten_numeric_metrics(dict(value), prefix=full_key))
            continue
        scalar = _scalar_number(value)
        if scalar is not None:
            flattened[full_key] = scalar
    return flattened


class SwanLabLogger:
    """The Controller's optional, single SwanLab experiment.

    The Controller constructs one instance for the full training run, calls
    :meth:`start` once, then calls :meth:`log_components` with telemetry it has
    received from every worker. Worker processes must never construct or start
    this class.

    ``start()`` is explicit so CPU unit tests do not import SwanLab or make
    network calls unless a real Controller starts an enabled logger.
    """

    def __init__(self, config: Mapping[str, object], *, pipeline: "TrainingPipeline") -> None:
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping.")

        self.config = copy.deepcopy(dict(config))
        self.run_cfg = _mapping(self.config.get("run"), "run")
        self.worker_namespaces = _pipeline_node_namespaces(pipeline)
        self.component_namespaces = self.worker_namespaces | _FRAMEWORK_NAMESPACES
        self.algorithm_namespace = _algorithm_namespace(
            self.config,
            reserved_namespaces=self.component_namespaces,
        )

        swanlab_cfg = _mapping(self.run_cfg.get("swanlab", {}), "run.swanlab")
        self.enabled = bool(swanlab_cfg.get("enabled", False))
        self.project = _optional_text(
            swanlab_cfg.get("project", "FrugalOPL-Official"), "run.swanlab.project"
        )
        self.workspace = _optional_text(swanlab_cfg.get("workspace", "qhe"), "run.swanlab.workspace")
        self.mode = _optional_text(swanlab_cfg.get("mode", "online"), "run.swanlab.mode")
        if self.mode not in {"online", "local", "offline", "disabled"}:
            raise ValueError("run.swanlab.mode must be online, local, offline, or disabled.")

        name = _optional_text(self.run_cfg.get("name"), "run.name")
        assert name is not None  # ``_optional_text`` only returns None for null input.
        self.run_name = name
        self.group = _optional_text(self.run_cfg.get("group"), "run.group")
        self._run: Any | None = None
        self._warned_log_failure = False

    @property
    def started(self) -> bool:
        return self._run is not None

    def start(self) -> None:
        """Initialize the one Controller-owned SwanLab experiment."""

        if not self.enabled or self._run is not None:
            return
        if not os.environ.get("SWANLAB_API_KEY"):
            raise RuntimeError(
                "SwanLab logging is enabled, but SWANLAB_API_KEY is absent from this process environment. "
                "Export it before starting the tmux training session."
            )
        try:
            import swanlab
        except ImportError as error:  # pragma: no cover - deployment dependency failure.
            raise RuntimeError(
                "SwanLab logging is enabled, but the 'swanlab' package is not installed in this Python environment."
            ) from error

        local_dir = self._local_log_dir()
        local_dir.mkdir(parents=True, exist_ok=True)
        self._run = swanlab.init(
            project=self.project,
            workspace=self.workspace,
            name=self.run_name,
            group=self.group,
            job_type="training",
            config=self.config,
            mode=self.mode,
            log_dir=str(local_dir),
        )
        print(
            f"[SwanLab] Started Controller-owned experiment project={self.project}, "
            f"group={self.group!r}, name={self.run_name}.",
            flush=True,
        )

    def log_components(
        self,
        components: Mapping[str, Mapping[str, object]],
        *,
        step: int,
    ) -> None:
        """Log finite scalar metrics beneath stable and algorithm namespaces.

        ``components`` keys must be a declared pipeline node ID, ``algorithm``,
        or ``evaluation``. The logical ``algorithm`` component is recorded
        beneath ``algorithm.algo_name``.
        """

        if not self.enabled or self._run is None:
            return
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("SwanLab metric step must be a non-negative integer.")

        payload: dict[str, float | int] = {}
        for raw_component, values in components.items():
            component = str(raw_component).strip().strip("/")
            if component not in self.component_namespaces:
                allowed = ", ".join(sorted(self.component_namespaces))
                raise ValueError(
                    f"Unknown SwanLab metrics namespace {component!r}; expected one of: {allowed}."
                )
            if not isinstance(values, Mapping):
                raise TypeError(f"SwanLab component {component!r} metrics must be a mapping.")
            namespace = self.algorithm_namespace if component == "algorithm" else component
            payload.update(_flatten_numeric_metrics(dict(values), prefix=namespace))
        if not payload:
            return
        try:
            self._run.log(payload, step=step)
        except Exception as error:  # Telemetry must not abort a valid optimization step.
            if not self._warned_log_failure:
                print(f"[SwanLab] WARNING: metric logging failed: {error}", flush=True)
                self._warned_log_failure = True

    def finish(self, *, state: SwanLabState = "success", error: BaseException | str | None = None) -> None:
        """Flush the Controller-owned experiment exactly once."""

        run, self._run = self._run, None
        if run is None:
            return
        detail = None if error is None else str(error)
        try:
            run.finish(state=state, error=detail)
        except Exception as finish_error:  # Never hide a Worker cleanup path behind telemetry.
            print(f"[SwanLab] WARNING: finish failed: {finish_error}", flush=True)

    def _local_log_dir(self) -> Path:
        storage = _mapping(self.run_cfg.get("storage", {}), "run.storage")
        configured_root = storage.get("output_root", "outputs")
        if not isinstance(configured_root, str) or not configured_root.strip():
            raise ValueError("run.storage.output_root must be a non-empty path string.")
        root = Path(configured_root).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        return root.resolve() / self.run_name / "swanlab"


__all__ = ["SwanLabLogger", "SwanLabState"]
