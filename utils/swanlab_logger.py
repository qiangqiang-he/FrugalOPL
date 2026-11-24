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
import json
import math
import os
import re
import tempfile
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

    def __init__(
        self,
        config: Mapping[str, object],
        *,
        pipeline: "TrainingPipeline",
        output_dir: str | Path | None = None,
    ) -> None:
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
        self._output_dir = None if output_dir is None else Path(output_dir).expanduser().resolve()
        self._resume_step = 0
        # RADO can reject multiple fresh batches at the same optimizer step.
        # A separate event axis preserves every decision instead of replacing
        # equal-step SwanLab points. The real optimizer step is always logged.
        self._event_steps = bool(swanlab_cfg.get('event_steps', False))
        self._event_step = -1
        if self._event_steps:
            event_path = self._local_log_dir().parent / 'swanlab_event_step.json'
            if event_path.is_file():
                self._event_step = int(json.loads(event_path.read_text(encoding='utf-8'))['event_step'])

    @property
    def started(self) -> bool:
        return self._run is not None

    @property
    def run_identity_path(self) -> Path:
        """The cloud run binding stored alongside this training run's state."""

        return self._local_log_dir().parent / "swanlab_run.json"

    @staticmethod
    def _validate_run_id(run_id: object) -> str:
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", run_id):
            raise ValueError("SwanLab run ID must be a non-empty identifier containing letters, digits, '_' or '-'.")
        return run_id

    def _run_identity(self, run_id: str) -> dict[str, object]:
        # Store only the cloud identity, never config, credentials, or environment.
        return {
            "version": 1,
            "workspace": self.workspace,
            "project": self.project,
            "run_name": self.run_name,
            "mode": "online",
            "run_id": run_id,
        }

    def _online_run_id(self, run: Any) -> str:
        run_id = self._validate_run_id(run.id)
        # Run.id reflects init settings in SwanLab 0.10.1. Run.path instead
        # comes from the backend's project path and experiment slug/CUID.
        cloud_path = getattr(run, "path", None)
        parts = cloud_path.strip("/").split("/") if isinstance(cloud_path, str) else []
        if (
            len(parts) != 3
            or parts[2] != run_id
            or (self.project is not None and parts[1] != self.project)
            or (self.workspace is not None and parts[0] != self.workspace)
        ):
            raise RuntimeError("SwanLab cloud run path does not match the requested experiment identity.")
        return run_id

    def _saved_run_id(self) -> str | None:
        path = self.run_identity_path
        try:
            with path.open(encoding="utf-8") as handle:
                identity = json.load(handle)
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            raise RuntimeError(f"Cannot read saved SwanLab run identity: {path}.") from None
        if not isinstance(identity, dict):
            raise RuntimeError(f"Invalid saved SwanLab run identity: {path}.")
        try:
            run_id = self._validate_run_id(identity.get("run_id"))
        except ValueError:
            raise RuntimeError(f"Invalid saved SwanLab run identity: {path}.") from None
        if identity != self._run_identity(run_id) or type(identity.get("version")) is not int:
            raise RuntimeError(f"Saved SwanLab run identity does not match this training run: {path}.")
        return run_id

    def seed_run_id(self, run_id: str) -> Path:
        """Bind an already-running online experiment without initializing SDK.

        This also persists the ID returned by first initialization. An existing
        binding is immutable: a conflicting or malformed record is never replaced.
        """

        if not self.enabled or self.mode != "online":
            raise ValueError("A SwanLab run identity requires enabled online logging.")
        run_id = self._validate_run_id(run_id)
        if self._run is not None and self._run.id != run_id:
            raise RuntimeError("Cannot bind a different SwanLab ID while a run is active.")
        saved_run_id = self._saved_run_id()
        if saved_run_id is not None:
            if saved_run_id != run_id:
                raise RuntimeError("Cannot replace an existing SwanLab run identity with another run.")
            return self.run_identity_path
        path = self.run_identity_path
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".swanlab-run-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(self._run_identity(run_id), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            try:
                # Publish a complete record atomically, without replacing a
                # conflicting binding from another Controller startup.
                os.link(temporary, path)
            except FileExistsError:
                if self._saved_run_id() != run_id:
                    raise RuntimeError("A conflicting SwanLab run identity was saved during initialization.") from None
        finally:
            os.unlink(temporary)
        return path

    def start(self) -> None:
        """Initialize or strictly resume the Controller-owned cloud experiment.

        Controller sets FRUGALOPL_RESUME_STEP before calling this method. A
        positive checkpoint step requires an existing matching cloud binding.
        """

        if not self.enabled or self._run is not None:
            return
        raw_resume_step = os.environ.get("FRUGALOPL_RESUME_STEP", "0")
        if not re.fullmatch(r"[0-9]+", raw_resume_step):
            raise ValueError("FRUGALOPL_RESUME_STEP must be a non-negative integer.")
        self._resume_step = int(raw_resume_step)
        saved_run_id = self._saved_run_id() if self.mode == "online" else None
        if self.mode == "online" and self._resume_step > 0 and saved_run_id is None:
            raise RuntimeError(
                "Cannot resume training without its saved SwanLab run identity; "
                "bind the existing experiment with seed_run_id() before restarting."
            )
        if self.mode == "online" and saved_run_id is None and os.environ.get("SWANLAB_RUN_ID"):
            raise RuntimeError(
                "A new training run must not inherit SWANLAB_RUN_ID; "
                "unset it or seed the verified existing experiment identity."
            )
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
        resume_options = {}
        if self.mode == "online":
            resume_options = {"resume": "never", "parallel": "none"}
            if saved_run_id is not None:
                resume_options.update(id=saved_run_id, resume="must")
        run = swanlab.init(
            project=self.project,
            workspace=self.workspace,
            name=self.run_name,
            group=self.group,
            job_type="training",
            config=self.config,
            mode=self.mode,
            log_dir=str(local_dir),
            **resume_options,
        )
        if self.mode == "online":
            try:
                run_id = self._online_run_id(run)
                if saved_run_id is not None and run_id != saved_run_id:
                    raise RuntimeError("SwanLab resumed a different experiment than the saved identity.")
                self.seed_run_id(run_id)
            except Exception:
                try:
                    run.finish(state="crashed", error="SwanLab cloud run identity could not be persisted.")
                except Exception:
                    pass
                raise
        self._run = run
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
        if step < self._resume_step:
            raise ValueError("SwanLab metric step must not precede the restored training step.")

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
        if self._event_steps:
            payload['controller/optimizer_step'] = step
            self._event_step += 1
            event_path = self._local_log_dir().parent / 'swanlab_event_step.json'
            event_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = event_path.with_suffix('.tmp')
            temporary.write_text(json.dumps({'event_step': self._event_step}), encoding='utf-8')
            os.replace(temporary, event_path)
            step = self._event_step
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
        if self._output_dir is not None:
            return self._output_dir / "swanlab"
        storage = _mapping(self.run_cfg.get("storage", {}), "run.storage")
        configured_root = storage.get("output_root", "outputs")
        if not isinstance(configured_root, str) or not configured_root.strip():
            raise ValueError("run.storage.output_root must be a non-empty path string.")
        root = Path(configured_root).expanduser()
        if not root.is_absolute():
            root = Path(__file__).resolve().parents[1] / root
        return root.resolve() / self.run_name / "swanlab"


__all__ = ["SwanLabLogger", "SwanLabState"]
