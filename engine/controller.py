"""Controller for FrugalOPL's asynchronous signal-pipeline training loop.

The Controller deliberately remains small.  It owns only dataset traversal,
Redis request/result routing, evaluation boundaries, and the lifetime of the
independent Worker processes.  Algorithm code still runs inside the Workers:
the Controller never renders a training prompt, computes a reward, or sees a
model tensor.

For a standard RLVR run the data flow is:

    Controller --train request--> Actor --trajectory--> Learner
    Controller <--metrics/result-- Learner

Intermediate Teacher nodes extend the same flow in configured order:

    Controller --train request--> Actor --trajectory--> Teacher* --signals--> Learner
    Controller <--metrics/result-- Learner

At an evaluation boundary the Controller drains all training requests, then
routes Actor-only requests through the separate evaluation Redis key.  This
makes an interval evaluation use the LoRA adapter published by the just
completed Learner step rather than racing future training rollouts.
"""

from __future__ import annotations

import argparse
import copy
import math
import multiprocessing as mp
import pickle
import random
import shutil
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from algorithm_plugins import plugin_factory_from_config
from algorithm_plugins.algorithm_base import AlgorithmPlugin
from engine.actor import ActorManager, load_merged_config
from engine.gpu_gate import GpuGateRegistry
from engine.learner import LearnerManager
from engine.pipeline import TrainingPipeline, resolve_training_pipeline, resolve_worker_config
from engine.teacher import TeacherManager
from utils.eval import (
    build_question_results,
    compute_metrics,
    evaluation_batch_limits,
    load_dataset_records,
    write_json,
)
from utils.swanlab_logger import SwanLabLogger


PluginFactory = Callable[[object], AlgorithmPlugin]


def _teacher_manager_items(value: object | None) -> tuple[tuple[str, object], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return tuple((str(node_id), manager) for node_id, manager in value.items())
    return (("teacher", value),)


def _mapping(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    return dict(value)


def _load_evaluation_datasets(value: object) -> dict[str, list[dict[str, object]]]:
    """Load named training-time evaluation datasets in configuration order."""

    if isinstance(value, Mapping):
        # Keep existing one-dataset training configurations usable while the
        # normal form evolves to a list of explicitly named datasets.
        entries: list[tuple[object | None, object]] = (
            [(None, value)] if "name" in value or "path" in value else list(value.items())
        )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        entries = [(None, item) for item in value]
    else:
        raise ValueError("controller.data.eval must be a dataset mapping or a list of dataset mappings.")

    datasets: dict[str, list[dict[str, object]]] = {}
    for index, (implicit_name, raw_dataset) in enumerate(entries):
        dataset = _mapping(raw_dataset, f"controller.data.eval[{index}]")
        name = dataset.get("name", implicit_name)
        if not isinstance(name, str) or not name.strip() or "/" in name:
            raise ValueError("Each controller.data.eval dataset needs a non-empty name without '/'.")
        name = name.strip()
        if name in datasets:
            raise ValueError(f"controller.data.eval contains duplicate dataset name {name!r}.")
        datasets[name] = load_dataset_records(dataset)
    if not datasets:
        raise ValueError("controller.data.eval may not be empty when evaluation is enabled.")
    return datasets


def _positive_int(value: object, path: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be an integer >= {minimum}.")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} must be an integer >= {minimum}.") from error
    if result < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}.")
    return result


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _json_number(value: object) -> object:
    """Preserve ordinary values while making non-finite metrics JSON-safe."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_number(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_number(item) for item in value]
    if isinstance(value, tuple):
        return [_json_number(item) for item in value]
    return value


class Controller:
    """Schedule one generic training run from a merged framework/algorithm config.

    The class is intentionally usable in CPU tests with a small Redis-shaped
    test double.  Production construction creates real ``ActorManager`` and
    ``LearnerManager`` instances, but model runtimes are imported only in
    their child processes.
    """

    def __init__(
        self,
        config: Mapping[str, object],
        *,
        plugin_factory: PluginFactory,
    ) -> None:
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping.")
        if not callable(plugin_factory):
            raise TypeError("plugin_factory must be callable.")

        self.config: dict[str, object] = copy.deepcopy(dict(config))
        self.pipeline: TrainingPipeline = resolve_training_pipeline(self.config)
        self.controller_node = self.pipeline.node_for_class("Controller")
        self.actor_node = self.pipeline.node_for_class("Actor")
        self.learner_node = self.pipeline.node_for_class("Learner")
        self.run_cfg = _mapping(self.config.get("run"), "run")
        self.controller_cfg = _mapping(
            self.config.get(self.controller_node.node_id), self.controller_node.node_id
        )
        self.actor_cfg = _mapping(self.config.get(self.actor_node.node_id), self.actor_node.node_id)
        self.plugin_factory = plugin_factory

        self.run_name, self.redis_prefix = self._run_name_and_redis()
        self.max_steps = _positive_int(self.controller_cfg.get("max_steps"), "controller.max_steps")
        self.train_prompt_batch_size = _positive_int(
            self.controller_cfg.get("train_prompt_batch_size"),
            "controller.train_prompt_batch_size",
        )
        self.max_inflight_batches = _positive_int(
            self.controller_cfg.get("max_inflight_batches", 1),
            "controller.max_inflight_batches",
        )

        engine_cfg = _mapping(self.actor_cfg.get("engine"), "actor.engine")
        self.actor_max_batches_per_turn = _positive_int(
            engine_cfg.get("max_batches_per_turn", 1),
            "actor.engine.max_batches_per_turn",
        )
        evaluation_cfg = _mapping(self.actor_cfg.get("evaluation", {}), "actor.evaluation")
        self.evaluation_enabled = bool(evaluation_cfg.get("enabled", True))
        self.evaluation_before_training = bool(evaluation_cfg.get("before_training", False))
        self.evaluation_interval_steps = int(evaluation_cfg.get("interval_steps", 0) or 0)
        if self.evaluation_interval_steps < 0:
            raise ValueError("actor.evaluation.interval_steps must be non-negative.")
        if self.evaluation_enabled and self.evaluation_interval_steps == 0 and not self.evaluation_before_training:
            # Evaluation is deliberately opt-in when no interval is supplied.
            self.evaluation_enabled = False
        self.eval_rollouts_per_batch, self.eval_rollouts_per_prompt, self.eval_prompts_per_batch = (
            evaluation_batch_limits(evaluation_cfg)
            if self.evaluation_enabled
            else (1, 1, 1)
        )
        self.save_eval_rollouts = bool(evaluation_cfg.get("save_rollouts", False))

        self.learner_cfg = _mapping(
            self.config.get(self.learner_node.node_id), self.learner_node.node_id
        )
        self.teacher_nodes = self.pipeline.teacher_nodes
        self.teacher_active = bool(self.teacher_nodes)
        self.gpu_gates = self._build_gpu_gates()

        data_cfg = _mapping(self.controller_cfg.get("data"), "controller.data")
        self.train_records = load_dataset_records(
            _mapping(data_cfg.get("train"), "controller.data.train"),
        )
        self.eval_datasets = _load_evaluation_datasets(data_cfg.get("eval")) if self.evaluation_enabled else {}
        # Retain the flattened view for lightweight callers that only need to
        # inspect the configured evaluation corpus. Runtime evaluation always
        # keeps dataset boundaries intact through ``eval_datasets``.
        self.eval_records = [record for records in self.eval_datasets.values() for record in records]
        self._train_order = list(range(len(self.train_records)))
        seed = int(self.run_cfg.get("seed", 0))
        random.Random(seed).shuffle(self._train_order)
        self._next_train_offset = 0

        self.output_dir = self._resolve_output_dir()
        self.train_history: list[dict[str, object]] = []
        self.evaluation_history: list[dict[str, object]] = []
        self._last_policy_version = 0
        # One Controller-owned SwanLab experiment receives detached Worker
        # telemetry through the normal Redis result path.  Workers never
        # initialize their own telemetry experiment.
        self._swanlab = SwanLabLogger(self.config, pipeline=self.pipeline)

    def _build_gpu_gates(self) -> GpuGateRegistry:
        """Create the run-local physical-GPU gate registry before spawning.

        Actor and Teacher entries each represent independent per-GPU Workers.
        Learner is the only collective Worker type; a multi-rank Learner that
        overlaps another Worker on just some of its GPUs is intentionally not
        supported by this lightweight per-GPU design, because it could let
        one rank enter a collective while another waits for a shared GPU.
        """

        def configured_gpu_ids(worker_name: str, cfg: Mapping[str, object]) -> list[object]:
            raw_gpu_ids = cfg.get("gpu_ids", [])
            if not isinstance(raw_gpu_ids, list):
                raise ValueError(f"{worker_name}.gpu_ids must be a list of GPU ids.")
            return list(raw_gpu_ids)

        worker_gpu_ids: dict[str, list[object]] = {
            self.actor_node.node_id: configured_gpu_ids(self.actor_node.node_id, self.actor_cfg),
            self.learner_node.node_id: configured_gpu_ids(self.learner_node.node_id, self.learner_cfg),
        }
        for node in self.teacher_nodes:
            teacher_cfg = resolve_worker_config(self.config, node.node_id)
            worker_gpu_ids[node.node_id] = configured_gpu_ids(node.node_id, teacher_cfg)
        learner_ids = set(worker_gpu_ids[self.learner_node.node_id])
        if len(learner_ids) > 1:
            non_learner_ids = set().union(
                *(
                    set(gpu_ids)
                    for worker_name, gpu_ids in worker_gpu_ids.items()
                    if worker_name != self.learner_node.node_id
                )
            )
            conflicting = learner_ids & non_learner_ids
            if conflicting:
                formatted = ", ".join(str(gpu_id) for gpu_id in sorted(conflicting))
                raise ValueError(
                    "A multi-GPU Learner may not share only part of its collective GPU set with another "
                    f"Worker (overlap: {formatted}). Use dedicated Learner GPUs for this configuration."
                )
        runtime_cfg = _mapping(self.run_cfg.get("runtime", {}), "run.runtime")
        context = mp.get_context(str(runtime_cfg.get("process_start_method", "spawn")))
        return GpuGateRegistry.from_worker_gpu_ids(worker_gpu_ids, ctx=context)

    def _run_name_and_redis(self) -> tuple[str, str]:
        name = self.run_cfg.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("run.name must be a non-empty string.")
        redis_cfg = _mapping(self.run_cfg.get("redis"), "run.redis")
        prefix = redis_cfg.get("key_prefix", "frugalopl")
        if not isinstance(prefix, str) or not prefix.strip():
            raise ValueError("run.redis.key_prefix must be a non-empty string.")
        return name.strip(), prefix.strip()

    def queue_keys(self) -> dict[str, str]:
        """Return the fixed, run-scoped keys shared with Actor and Learner."""

        root = f"{self.redis_prefix}:{self.run_name}"
        return {
            "root": root,
            "actor_train": f"{root}:actor:train",
            "actor_eval": f"{root}:actor:eval",
            "teacher_train": f"{root}:teacher:train",
            "learner_train": f"{root}:learner:train",
            "train_results": f"{root}:controller:train_results",
            "eval_results": f"{root}:controller:eval_results",
            "events": f"{root}:controller:events",
        }

    def _resolve_output_dir(self) -> Path:
        storage_cfg = _mapping(self.run_cfg.get("storage", {}), "run.storage")
        configured = storage_cfg.get("output_root", "outputs")
        if not isinstance(configured, str) or not configured.strip():
            raise ValueError("run.storage.output_root must be a non-empty path string.")
        root = Path(configured).expanduser()
        if not root.is_absolute():
            root = _project_root() / root
        return root.resolve() / self.run_name

    def _lora_adapter_root(self) -> Path:
        """Return the Learner's published-adapter directory for this run.

        This deliberately mirrors the default used by ``Learner`` without
        importing that module's private helper.  An explicit path remains
        supported for advanced deployments, while the normal case lives under
        this run's output directory and is therefore safe to snapshot at an
        evaluation boundary.
        """

        storage_cfg = _mapping(self.run_cfg.get("storage", {}), "run.storage")
        configured = storage_cfg.get("lora_adapter_root")
        if configured is None:
            return self.output_dir / "tmp" / "_tmp_lora"
        if not isinstance(configured, str) or not configured.strip():
            raise ValueError("run.storage.lora_adapter_root must be a non-empty path string or null.")
        root = Path(configured).expanduser()
        if not root.is_absolute():
            root = _project_root() / root
        return root.resolve()

    @staticmethod
    def _validate_complete_peft_adapter(path: Path, *, label: str) -> None:
        """Require the portable PEFT files needed by an Actor LoRA request."""

        if not path.is_dir():
            raise RuntimeError(f"{label} is not a directory: {path}")
        required = ("adapter_config.json", "adapter_model.safetensors")
        missing = [name for name in required if not (path / name).is_file()]
        if missing:
            listed = ", ".join(missing)
            raise RuntimeError(
                f"{label} is not a complete PEFT adapter at {path}; missing {listed}. "
                "Expected a direct adapter root containing adapter_config.json and "
                "adapter_model.safetensors."
            )

    def _published_adapter_source(self, history_item: Mapping[str, object]) -> Path:
        """Find the fully published LoRA corresponding to a completed update."""

        learner_metrics = self._worker_component_metrics(
            history_item,
            component=self.learner_node.node_id,
            metric_field="learner_metrics",
        )
        reported = learner_metrics.get("lora_path")
        if isinstance(reported, str) and reported.strip():
            return Path(reported).expanduser().resolve()

        policy_version = history_item.get("policy_version")
        if isinstance(policy_version, bool) or not isinstance(policy_version, int) or policy_version < 0:
            raise RuntimeError(
                "Controller cannot snapshot an evaluation adapter because the completed Learner result "
                "has no valid policy_version."
            )
        return self._lora_adapter_root() / f"step_{policy_version}" / "adapter_lora"

    def _snapshot_evaluation_adapter(
        self,
        *,
        completed_steps: int,
        history_item: Mapping[str, object],
    ) -> Path:
        """Copy the boundary policy into an immutable, directly loadable adapter root.

        The Actor and Learner continue to use their normal publication path;
        this method only reads it after the Controller has drained the training
        boundary.  A completed destination is accepted as an idempotent retry,
        but an incomplete pre-existing destination is always surfaced rather
        than silently reused or overwritten.
        """

        if completed_steps <= 0:
            raise ValueError("Evaluation adapter snapshots are only valid after a positive training step.")

        source = self._published_adapter_source(history_item)
        self._validate_complete_peft_adapter(source, label="Published Learner LoRA adapter")

        destination = self.output_dir / "eval_adapters" / f"step_{completed_steps}"
        if destination.exists():
            self._validate_complete_peft_adapter(destination, label="Existing evaluation adapter snapshot")
            return destination

        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.parent / f".step_{completed_steps}.copy-{uuid.uuid4().hex}"
        try:
            # Copy the *contents* of adapter_lora into step_<N>.  The resulting
            # path can therefore be passed directly as actor.lora.path.
            shutil.copytree(source, staging)
            self._validate_complete_peft_adapter(staging, label="Staged evaluation adapter snapshot")
            try:
                staging.replace(destination)
            except FileExistsError:
                # A retry or a second Controller that won this boundary first
                # is safe only when the already-visible snapshot is complete.
                self._validate_complete_peft_adapter(destination, label="Existing evaluation adapter snapshot")
        except Exception as error:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            if isinstance(error, RuntimeError):
                raise
            raise RuntimeError(
                f"Controller could not snapshot evaluation adapter at training step {completed_steps} "
                f"from {source} to {destination}: {error}"
            ) from error

        if not destination.exists():  # pragma: no cover - defensive filesystem invariant.
            raise RuntimeError(
                f"Controller did not materialize evaluation adapter snapshot at training step {completed_steps}: "
                f"{destination}"
            )
        self._validate_complete_peft_adapter(destination, label="Evaluation adapter snapshot")
        return destination

    def _redis_client(self) -> object:
        try:
            import redis
        except ImportError as error:  # pragma: no cover - deployment dependency failure.
            raise RuntimeError("redis-py is required to run the Controller.") from error
        redis_cfg = _mapping(self.run_cfg.get("redis"), "run.redis")
        url = redis_cfg.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ValueError("run.redis.url must be a non-empty string.")
        return redis.Redis.from_url(url, decode_responses=False)

    def _clear_run_keys(self, client: object) -> None:
        redis_cfg = _mapping(self.run_cfg.get("redis"), "run.redis")
        if not bool(redis_cfg.get("clear_run_keys_on_start", False)):
            return
        scan_iter = getattr(client, "scan_iter", None)
        delete = getattr(client, "delete", None)
        if not callable(scan_iter) or not callable(delete):
            raise TypeError("Controller Redis client must provide scan_iter() and delete().")
        prefix = f"{self.queue_keys()['root']}:"
        owned: list[object] = []
        for key in scan_iter(match=f"{prefix}*"):
            text = key.decode() if isinstance(key, bytes) else str(key)
            if text.startswith(prefix):
                owned.append(key)
        if owned:
            delete(*owned)
            print(f"[Controller] Cleared {len(owned)} stale Redis key(s) for {self.run_name}.", flush=True)

    def _next_train_records(self) -> list[dict[str, object]]:
        """Take a deterministic prompt batch, cycling only after dataset exhaustion."""

        selected: list[dict[str, object]] = []
        for _ in range(self.train_prompt_batch_size):
            record_index = self._train_order[self._next_train_offset % len(self._train_order)]
            selected.append(copy.deepcopy(self.train_records[record_index]))
            self._next_train_offset += 1
        return selected

    @staticmethod
    def _records_to_signals(records: Sequence[Mapping[str, object]]) -> tuple[list[str], list[object]]:
        questions: list[str] = []
        answers: list[object] = []
        for record_index, record in enumerate(records):
            question = record.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Dataset record {record_index} must have a non-empty string question.")
            if "answer" not in record or record["answer"] is None:
                raise ValueError(f"Dataset record {record_index} must have a non-null answer.")
            questions.append(question)
            answers.append(record["answer"])
        return questions, answers

    def _push_training_group(
        self,
        client: object,
        pending: dict[str, dict[str, object]],
        *,
        count: int,
        issued: int,
    ) -> int:
        """Atomically queue one bounded Actor execution turn."""

        if count < 1:
            return issued
        serialized: list[bytes] = []
        new_entries: list[tuple[str, dict[str, object]]] = []
        for _ in range(count):
            records = self._next_train_records()
            questions, answers = self._records_to_signals(records)
            sequence = issued + 1
            batch_id = f"train-{sequence:06d}"
            signals: dict[str, object] = {
                "batch_id": batch_id,
                "policy_version": self._last_policy_version,
                "questions": questions,
                "answers": answers,
                # Preserve prompt-level metadata (for example privileged
                # solutions) without teaching the framework algorithm fields.
                "records": copy.deepcopy(records),
            }
            serialized.append(pickle.dumps(signals, protocol=pickle.HIGHEST_PROTOCOL))
            new_entries.append((batch_id, {"sequence": sequence, "record_count": len(records)}))
            issued += 1
        lpush = getattr(client, "lpush", None)
        if not callable(lpush):
            raise TypeError("Controller Redis client must provide lpush().")
        # A variadic LPUSH is atomic.  BRPOP retrieves the first payload and
        # Actor's immediate RPOP retrieves the next one, preserving the
        # logical order while allowing one bounded execution turn to coalesce.
        # Record immediately before the actual Redis send so the end-to-end
        # number includes queueing, every Worker stage, and the result route.
        dispatched_at = time.monotonic()
        lpush(self.queue_keys()["actor_train"], *serialized)
        for _batch_id, entry in new_entries:
            entry["dispatch_monotonic"] = dispatched_at
        pending.update(dict(new_entries))
        first = new_entries[0][0]
        last = new_entries[-1][0]
        print(f"[Controller] Dispatched {first}..{last} ({len(pending)} training batch(es) in flight).", flush=True)
        return issued

    def _pop_message(
        self,
        client: object,
        *,
        result_key: str,
        actor_manager: ActorManager,
        learner_manager: LearnerManager,
        teacher_manager: TeacherManager | Mapping[str, TeacherManager] | None = None,
    ) -> dict[str, object]:
        """Wait for a result, surfacing Worker events rather than timing out silently."""

        brpop = getattr(client, "brpop", None)
        if not callable(brpop):
            raise TypeError("Controller Redis client must provide brpop().")
        keys = self.queue_keys()
        while True:
            self._assert_workers_alive(actor_manager, learner_manager, teacher_manager)
            received = brpop([result_key, keys["events"]], timeout=1)
            if received is None:
                continue
            raw_key, raw_payload = received
            key = raw_key.decode() if isinstance(raw_key, bytes) else str(raw_key)
            try:
                decoded = pickle.loads(raw_payload)
            except Exception as error:
                raise RuntimeError(f"Controller could not decode Redis payload from {key}: {error}") from error
            if not isinstance(decoded, Mapping):
                raise RuntimeError(f"Controller received a non-mapping Redis payload from {key}.")
            message = dict(decoded)
            if key == keys["events"]:
                source = message.get("source", "worker")
                detail = message.get("message", message)
                raise RuntimeError(f"{source} reported a runtime event: {detail}")
            if key != result_key:
                raise RuntimeError(f"Controller received an unexpected Redis key: {key}")
            return message

    @staticmethod
    def _assert_workers_alive(
        actor_manager: ActorManager,
        learner_manager: LearnerManager,
        teacher_manager: TeacherManager | Mapping[str, TeacherManager] | None = None,
    ) -> None:
        def terminal_failure(manager: object) -> str | None:
            """Read a Manager's fail-fast state without constraining test doubles."""

            value = getattr(manager, "fatal_error", None)
            if value is None:
                return None
            if isinstance(value, str):
                return value or None
            return str(value)

        def dead(processes: Mapping[object, object]) -> list[tuple[object, object]]:
            failures: list[tuple[object, object]] = []
            for identifier, process in processes.items():
                is_alive = getattr(process, "is_alive", None)
                if callable(is_alive) and not is_alive():
                    failures.append((identifier, getattr(process, "exitcode", None)))
            return failures

        actor_terminal_failure = terminal_failure(actor_manager)
        learner_terminal_failure = terminal_failure(learner_manager)
        teacher_terminal_failures = {
            node_id: failure
            for node_id, manager in _teacher_manager_items(teacher_manager)
            if (failure := terminal_failure(manager)) is not None
        }
        if actor_terminal_failure or learner_terminal_failure or teacher_terminal_failures:
            raise RuntimeError(
                "A Controller-owned Worker supervisor entered fail-fast state: "
                f"actor={actor_terminal_failure!r}, teachers={teacher_terminal_failures!r}, "
                f"learner={learner_terminal_failure!r}."
            )

        actor_failures = dead(actor_manager.processes)
        learner_failures = dead(learner_manager.processes)
        teacher_failures = {
            node_id: dead(getattr(manager, "processes", {}))
            for node_id, manager in _teacher_manager_items(teacher_manager)
        }
        teacher_failures = {node_id: failures for node_id, failures in teacher_failures.items() if failures}
        if actor_failures or learner_failures or teacher_failures:
            raise RuntimeError(
                "A Controller-owned Worker exited before producing its expected result: "
                f"actor={actor_failures}, teachers={teacher_failures}, learner={learner_failures}."
            )

    @staticmethod
    def _worker_component_metrics(
        message: Mapping[str, object],
        *,
        component: str,
        metric_field: str,
    ) -> dict[str, object]:
        """Merge a Worker's detached metrics and its framework timing.

        Actor and Teacher attach role-owned metrics to their ordinary Redis
        message. Every Worker also appends timing below
        ``timing.workers.<component>``. The Controller is deliberately the
        only SwanLab writer, so this helper converts that transport envelope
        back into one stable component namespace without exposing Redis to the
        telemetry layer.
        """

        metrics: dict[str, object] = {}
        raw_metrics = message.get(metric_field)
        if isinstance(raw_metrics, Mapping):
            metrics.update(dict(raw_metrics))
        timing = message.get("timing")
        if isinstance(timing, Mapping):
            workers = timing.get("workers")
            if isinstance(workers, Mapping):
                worker_timing = workers.get(component)
                if isinstance(worker_timing, Mapping):
                    metrics.update(dict(worker_timing))
        return metrics

    def _run_evaluation(
        self,
        client: object,
        *,
        completed_steps: int,
        actor_manager: ActorManager,
        learner_manager: LearnerManager,
        teacher_manager: TeacherManager | Mapping[str, TeacherManager] | None = None,
        adapter_snapshot: Path | None = None,
    ) -> dict[str, object]:
        """Run one full Actor-only evaluation after a drained training boundary."""

        if not self.evaluation_enabled:
            return {}
        started = time.time()
        keys = self.queue_keys()
        dataset_summaries: dict[str, object] = {}
        detail_datasets: dict[str, object] = {}

        for dataset_name, dataset_records in self.eval_datasets.items():
            evaluation_id = f"eval-step-{completed_steps:06d}-{dataset_name}"
            question_results: list[dict[str, object]] = []
            for batch_index, start in enumerate(
                range(0, len(dataset_records), self.eval_prompts_per_batch)
            ):
                records = dataset_records[start : start + self.eval_prompts_per_batch]
                questions, answers = self._records_to_signals(records)
                signals: dict[str, object] = {
                    "evaluation_id": evaluation_id,
                    "evaluation_batch_index": batch_index,
                    "policy_version": self._last_policy_version,
                    "questions": questions,
                    "answers": answers,
                    "records": copy.deepcopy(records),
                }
                lpush = getattr(client, "lpush", None)
                if not callable(lpush):
                    raise TypeError("Controller Redis client must provide lpush().")
                # Evaluation is intentionally one request at a time. Therefore
                # every vLLM call obeys the project-wide 64-rollout maximum.
                lpush(keys["actor_eval"], pickle.dumps(signals, protocol=pickle.HIGHEST_PROTOCOL))
                result = self._pop_message(
                    client,
                    result_key=keys["eval_results"],
                    actor_manager=actor_manager,
                    learner_manager=learner_manager,
                    teacher_manager=teacher_manager,
                )
                if result.get("evaluation_id") != evaluation_id:
                    raise RuntimeError(
                        f"Expected evaluation result {evaluation_id!r}, got {result.get('evaluation_id')!r}."
                    )
                if int(result.get("evaluation_batch_index", -1)) != batch_index:
                    raise RuntimeError("Controller received an evaluation result for the wrong batch index.")
                trajectory = result.get("traj")
                if not isinstance(trajectory, Mapping):
                    raise RuntimeError("Actor evaluation result does not contain a trajectory mapping.")
                question_results.extend(
                    build_question_results(
                        records,
                        trajectory,
                        rollouts_per_prompt=self.eval_rollouts_per_prompt,
                        save_rollouts=self.save_eval_rollouts,
                        start_index=start,
                    )
                )

            question_results.sort(key=lambda item: int(item["record_index"]))
            raw_metrics = compute_metrics(question_results, rollouts_per_prompt=self.eval_rollouts_per_prompt)
            avg_at_k = f"avg_at_{self.eval_rollouts_per_prompt}"
            pass_at_k = f"pass_at_{self.eval_rollouts_per_prompt}"
            # Keep the two metrics researchers compare first at the front of
            # each dataset namespace, and leave all bookkeeping out of it.
            metrics = {
                avg_at_k: raw_metrics[avg_at_k],
                pass_at_k: raw_metrics[pass_at_k],
                "avg_response_length": raw_metrics["average_response_length"],
                "truncation_rate": raw_metrics["truncation_rate"],
                "correct_response_count": raw_metrics["correct_response_count"],
                "incorrect_response_count": raw_metrics["incorrect_response_count"],
            }
            dataset_summaries[dataset_name] = {"metrics": metrics}
            detail_datasets[dataset_name] = {"questions": question_results}

        result: dict[str, object] = {
            "step": completed_steps,
            "policy_version": self._last_policy_version,
            "elapsed_seconds": time.time() - started,
            "datasets": dataset_summaries,
        }
        if adapter_snapshot is not None:
            # Keep the immutable artifact path in local JSON. SwanLab records
            # the numeric step below, while the path remains useful for a
            # later standalone Actor evaluation or exact reproduction.
            result["adapter_snapshot"] = str(adapter_snapshot)
            result["adapter_snapshot_step"] = completed_steps
        self.evaluation_history.append(result)
        self._log_evaluation(result)
        details = {
            "run_name": self.run_name,
            "step": completed_steps,
            "rollouts_per_prompt": self.eval_rollouts_per_prompt,
            "datasets": detail_datasets,
        }
        if adapter_snapshot is not None:
            details["adapter_snapshot"] = str(adapter_snapshot)
        write_json(self.output_dir / "evaluations" / f"step_{completed_steps:06d}_summary.json", result)
        write_json(self.output_dir / "evaluations" / f"step_{completed_steps:06d}_details.json", details)
        for dataset_name, dataset in dataset_summaries.items():
            metrics = dataset["metrics"]
            assert isinstance(metrics, Mapping)
            print(
                f"[Controller] Evaluation step={completed_steps}, dataset={dataset_name}: "
                f"Avg@{self.eval_rollouts_per_prompt}={float(metrics[avg_at_k]):.4f}, "
                f"Pass@{self.eval_rollouts_per_prompt}={float(metrics[pass_at_k]):.4f}, "
                f"avg_len={float(metrics['avg_response_length']):.2f}, "
                f"truncation_rate={float(metrics['truncation_rate']):.4f}.",
                flush=True,
            )
        return result

    def _completed_response_tokens(self, current_item: Mapping[str, object]) -> int:
        """Sum generated response tokens across all completed logical batches."""

        entries: list[Mapping[str, object]] = list(self.train_history)
        if not any(entry is current_item for entry in entries):
            # This keeps the pure logging helper useful in CPU tests and for
            # external Controller-like callers that log before appending.
            entries.append(current_item)

        total = 0
        for entry in entries:
            actor_metrics = self._worker_component_metrics(
                entry,
                component=self.actor_node.node_id,
                metric_field="actor_metrics",
            )
            total += int(actor_metrics["batch_response_tokens_total"])
        return total

    def _global_progress_metrics(
        self,
        history_item: Mapping[str, object],
        *,
        elapsed_seconds: float,
    ) -> dict[str, object]:
        """Return Controller-global metrics at one completed optimizer step."""

        completed_step = int(history_item["completed_step"])
        elapsed = max(0.0, float(elapsed_seconds))
        response_tokens = self._completed_response_tokens(history_item)
        return {
            "elapsed_seconds": elapsed,
            "avg_batch_time_seconds": elapsed / completed_step,
            "response_tokens_total": response_tokens,
            "avg_batch_response_tokens": response_tokens / completed_step,
        }

    def _write_progress(self, *, started: float, completed_steps: int) -> None:
        summary = {
            "run_name": self.run_name,
            "configured_max_steps": self.max_steps,
            "completed_steps": completed_steps,
            "elapsed_seconds": time.time() - started,
            "train": self.train_history,
            "evaluations": self.evaluation_history,
        }
        write_json(self.output_dir / "training_summary.json", _json_number(summary))

    def _log_completed_training_batch(
        self,
        history_item: Mapping[str, object],
        *,
        started: float,
    ) -> None:
        """Record one complete training update in the single experiment.

        The Controller receives all detached worker telemetry only after the
        Learner has completed the logical optimizer update. Emitting every
        configured Worker namespace together at that boundary gives every
        chart the same global-step axis without making any Worker an
        independent experiment writer.
        """

        completed_step = int(history_item["completed_step"])
        actor_metrics = self._worker_component_metrics(
            history_item,
            component=self.actor_node.node_id,
            metric_field="actor_metrics",
        )
        controller_metrics = {
            key: actor_metrics[key]
            for key in (
                "batch_prompt_tokens_total",
                "batch_prompt_tokens_mean",
                "batch_response_tokens_total",
                "batch_response_tokens_mean",
            )
        }
        persisted_progress = history_item.get("controller_metrics")
        if isinstance(persisted_progress, Mapping):
            progress_metrics = persisted_progress
        else:
            progress_metrics = self._global_progress_metrics(
                history_item,
                elapsed_seconds=max(0.0, time.time() - started),
            )
        controller_metrics.update(
            {
                key: progress_metrics[key]
                for key in (
                    "elapsed_seconds",
                    "avg_batch_time_seconds",
                    "response_tokens_total",
                    "avg_batch_response_tokens",
                )
            }
        )

        raw_algorithm = history_item.get("algorithm_metrics")
        if not isinstance(raw_algorithm, Mapping):
            # Legacy Learner results predate the explicit envelope. Keeping
            # this fallback makes Controller-compatible plugin tests useful.
            raw_algorithm = history_item.get("metrics", {})
        algorithm_metrics = (
            {key: value for key, value in raw_algorithm.items() if key != "objective"}
            if isinstance(raw_algorithm, Mapping)
            else {}
        )

        # Learner mirrors algorithm metrics by design, then appends its own
        # optimizer and execution telemetry (learning rate, grad norm, etc.).
        learner_metrics = dict(algorithm_metrics)
        learner_metrics.update(
            self._worker_component_metrics(
                history_item,
                component=self.learner_node.node_id,
                metric_field="learner_metrics",
            )
        )
        learner_metrics.pop("objective", None)
        components: dict[str, Mapping[str, object]] = {
            self.controller_node.node_id: controller_metrics,
            self.learner_node.node_id: learner_metrics,
            "algorithm": algorithm_metrics,
        }
        if actor_metrics:
            components[self.actor_node.node_id] = actor_metrics
        for node in self.teacher_nodes:
            teacher_metrics = self._worker_component_metrics(
                history_item,
                component=node.node_id,
                metric_field=f"{node.node_id}_metrics",
            )
            if teacher_metrics:
                components[node.node_id] = teacher_metrics
        self._swanlab.log_components(components, step=completed_step)

    def _log_evaluation(self, result: Mapping[str, object]) -> None:
        """Record only the global and per-dataset evaluation results."""

        step = result.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise RuntimeError("Controller evaluation telemetry requires a non-negative integer step.")
        values: dict[str, object] = {}
        elapsed = result.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
            values["elapsed_seconds"] = elapsed
        adapter_snapshot_step = result.get("adapter_snapshot_step")
        if isinstance(adapter_snapshot_step, int) and not isinstance(adapter_snapshot_step, bool):
            values["adapter_snapshot_step"] = adapter_snapshot_step
        datasets = result.get("datasets")
        if isinstance(datasets, Mapping):
            for dataset_name, dataset in datasets.items():
                if not isinstance(dataset_name, str) or not isinstance(dataset, Mapping):
                    continue
                raw_metrics = dataset.get("metrics")
                if not isinstance(raw_metrics, Mapping):
                    continue
                avg_key = next(
                    (key for key in raw_metrics if isinstance(key, str) and key.startswith("avg_at_")),
                    None,
                )
                if avg_key is None:
                    continue
                pass_key = f"pass_at_{avg_key.removeprefix('avg_at_')}"
                metric_keys = (
                    avg_key,
                    pass_key,
                    "avg_response_length",
                    "correct_response_count",
                    "incorrect_response_count",
                )
                if all(key in raw_metrics for key in metric_keys):
                    values[dataset_name] = {key: raw_metrics[key] for key in metric_keys}
                    if "truncation_rate" in raw_metrics:
                        values[dataset_name]["truncation_rate"] = raw_metrics["truncation_rate"]
        self._swanlab.log_components({"evaluation": values}, step=step)

    @staticmethod
    def _shutdown_managers(
        *,
        actor_manager: ActorManager,
        learner_manager: LearnerManager,
        teacher_manager: TeacherManager | Mapping[str, TeacherManager] | None,
    ) -> None:
        """Stop every Worker role without killing a shared-GPU permit holder.

        All current Managers expose a two-phase lifecycle: ``request_stop``
        merely sets their event, while ``finalize_stop`` joins and only then
        escalates stubborn children.  Broadcasting the first phase before any
        join matters on one-card runs.  For example, Actor may be tearing down
        vLLM under the physical-GPU gate while Learner must wait for that gate
        before it can destroy its process group.  Setting both events first
        lets both normal cleanup paths make progress.

        The small compatibility fallback keeps Controller CPU test doubles and
        third-party Manager-like implementations usable: an object without the
        two-phase hooks still receives its traditional ``stop()`` call during
        finalization.
        """

        managers: list[object] = [actor_manager]
        managers.extend(manager for _node_id, manager in _teacher_manager_items(teacher_manager))
        managers.append(learner_manager)

        errors: list[BaseException] = []
        for manager in managers:
            request_stop = getattr(manager, "request_stop", None)
            if not callable(request_stop):
                continue
            try:
                request_stop()
            except BaseException as error:
                errors.append(error)

        # Do not abandon later Worker roles merely because one Manager's
        # cleanup raised.  Every owned child still needs a chance to release
        # GPU/NCCL resources and be reaped.
        for manager in managers:
            finalize_stop = getattr(manager, "finalize_stop", None)
            stop = finalize_stop if callable(finalize_stop) else getattr(manager, "stop", None)
            if not callable(stop):
                errors.append(TypeError(f"{type(manager).__name__} exposes neither finalize_stop() nor stop()."))
                continue
            try:
                stop()
            except BaseException as error:
                errors.append(error)

        if errors:
            raise RuntimeError("One or more Worker Managers failed during shutdown.") from errors[0]

    def run(self) -> dict[str, object]:
        """Launch Workers, train through ``max_steps``, and evaluate at boundaries."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.output_dir / "merged_config.json", self.config)
        # Credentials are read only from the already-exported process
        # environment.  Start before Workers so an invalid SwanLab setup
        # fails before any GPU model is loaded.
        self._swanlab.start()
        client = self._redis_client()
        ping = getattr(client, "ping", None)
        if not callable(ping):
            raise TypeError("Controller Redis client must provide ping().")
        ping()
        self._clear_run_keys(client)

        learner_manager = LearnerManager(
            self.config,
            plugin_factory=self.plugin_factory,
            gpu_gates=self.gpu_gates,
        )
        teacher_manager = {
            node.node_id: TeacherManager(
                self.config,
                plugin_factory=self.plugin_factory,
                node_id=node.node_id,
                gpu_gates=self.gpu_gates,
            )
            for node in self.teacher_nodes
        }
        actor_manager = ActorManager(
            self.config,
            plugin_factory=self.plugin_factory,
            gpu_gates=self.gpu_gates,
        )
        supervisors: list[threading.Thread] = []
        started = time.time()
        completed_steps = 0
        issued_steps = 0
        pending: dict[str, dict[str, object]] = {}
        run_error: BaseException | None = None
        try:
            # Learner first means an Actor can hand off its initial trajectory
            # as soon as vLLM becomes ready instead of filling an orphan queue.
            learner_manager.start()
            # On a shared GPU, loading multiple model runtimes concurrently
            # creates an avoidable transient peak.  Wait for each complete
            # Worker type before the next one begins model initialization.
            learner_manager.wait_until_ready()
            learner_supervisor = threading.Thread(
                target=learner_manager.join,
                name="FrugalOPL-LearnerSupervisor",
                daemon=True,
            )
            learner_supervisor.start()
            supervisors.append(learner_supervisor)
            for node_id, manager in reversed(tuple(teacher_manager.items())):
                # Start from the downstream end so every upstream queue has a
                # ready consumer before Actor begins generation.  This launch
                # order is independent of the configured signal-flow order.
                manager.start()
                manager.wait_until_ready()
                teacher_supervisor = threading.Thread(
                    target=manager.join,
                    name=f"FrugalOPL-{node_id}-Supervisor",
                    daemon=True,
                )
                teacher_supervisor.start()
                supervisors.append(teacher_supervisor)
            actor_manager.start()
            actor_manager.wait_until_ready()
            actor_supervisor = threading.Thread(
                target=actor_manager.join,
                name="FrugalOPL-ActorSupervisor",
                daemon=True,
            )
            actor_supervisor.start()
            supervisors.append(actor_supervisor)
            print(
                f"[Controller] Started run={self.run_name}, steps={self.max_steps}, "
                f"train_batch={self.train_prompt_batch_size}, "
                f"actor_turn={self.actor_max_batches_per_turn}, "
                f"shared_gpus={list(self.gpu_gates.shared_gpu_ids)}.",
                flush=True,
            )

            if self.evaluation_enabled and self.evaluation_before_training:
                self._run_evaluation(
                    client,
                    completed_steps=0,
                    actor_manager=actor_manager,
                    learner_manager=learner_manager,
                    teacher_manager=teacher_manager,
                )

            while completed_steps < self.max_steps:
                next_boundary = self.max_steps
                if self.evaluation_enabled and self.evaluation_interval_steps:
                    next_boundary = min(
                        self.max_steps,
                        ((completed_steps // self.evaluation_interval_steps) + 1) * self.evaluation_interval_steps,
                    )

                # Keep dispatches inside the current interval.  A Worker turn
                # may process *up to* its configured width, so a partially
                # filled turn is valid whenever one in-flight slot opens.
                # This is what lets distinct GPUs pipeline through Redis
                # without introducing Controller-owned per-batch permission.
                remaining_to_boundary = next_boundary - issued_steps
                capacity = self.max_inflight_batches - len(pending)
                if remaining_to_boundary > 0 and capacity > 0:
                    group_size = min(self.actor_max_batches_per_turn, capacity, remaining_to_boundary)
                    issued_steps = self._push_training_group(
                        client,
                        pending,
                        count=group_size,
                        issued=issued_steps,
                    )

                if not pending:
                    # A positive configured limit should always have made
                    # progress above.  Keep the explicit failure rather than
                    # silently spinning on a scheduling regression.
                    if issued_steps < next_boundary:
                        raise RuntimeError("Controller could not dispatch a training batch before its boundary.")
                    raise RuntimeError("Controller reached an empty training queue before its boundary.")

                result = self._pop_message(
                    client,
                    result_key=self.queue_keys()["train_results"],
                    actor_manager=actor_manager,
                    learner_manager=learner_manager,
                    teacher_manager=teacher_manager,
                )
                batch_id = result.get("batch_id")
                if not isinstance(batch_id, str) or batch_id not in pending:
                    raise RuntimeError(f"Controller received an unknown training result: {batch_id!r}.")
                pending_info = pending.pop(batch_id)
                result_timing = result.get("timing", {})
                timing = dict(result_timing) if isinstance(result_timing, Mapping) else {}
                dispatched_at = pending_info.get("dispatch_monotonic")
                if isinstance(dispatched_at, (int, float)) and not isinstance(dispatched_at, bool):
                    timing["end_to_end_seconds"] = max(0.0, time.monotonic() - float(dispatched_at))
                completed_steps += 1
                global_step = result.get("global_step")
                if isinstance(global_step, bool):
                    raise RuntimeError("Learner global_step must be an integer, not bool.")
                try:
                    self._last_policy_version = int(global_step)
                except (TypeError, ValueError) as error:
                    raise RuntimeError("Learner result does not contain an integer global_step.") from error
                history_item = {
                    "completed_step": completed_steps,
                    "batch_id": batch_id,
                    "policy_version": self._last_policy_version,
                    "dispatch": pending_info,
                    "metrics": result.get("metrics", {}),
                    "algorithm_metrics": result.get("algorithm_metrics", result.get("metrics", {})),
                    "learner_metrics": result.get("learner_metrics", {}),
                    "timing": timing,
                }
                metric_keys = ["actor_metrics"]
                metric_keys.extend(f"{node.node_id}_metrics" for node in self.teacher_nodes)
                for metric_key in metric_keys:
                    if metric_key in result:
                        history_item[metric_key] = result[metric_key]
                self.train_history.append(history_item)
                completion_elapsed = max(0.0, time.time() - started)
                history_item["controller_metrics"] = self._global_progress_metrics(
                    history_item,
                    elapsed_seconds=completion_elapsed,
                )
                end_to_end = timing.get("end_to_end_seconds")
                timing_text = (
                    f", end_to_end={float(end_to_end):.2f}s"
                    if isinstance(end_to_end, (int, float)) and not isinstance(end_to_end, bool)
                    else ""
                )
                progress_metrics = history_item["controller_metrics"]
                assert isinstance(progress_metrics, Mapping)  # local construction invariant.
                print(
                    f"[Controller] Completed {completed_steps}/{self.max_steps}: "
                    f"{batch_id}, policy_step={self._last_policy_version}{timing_text}, "
                    f"avg_batch_time_seconds={float(progress_metrics['avg_batch_time_seconds']):.2f}s, "
                    f"avg_batch_response_tokens={float(progress_metrics['avg_batch_response_tokens']):.2f}.",
                    flush=True,
                )
                self._log_completed_training_batch(
                    history_item,
                    started=started,
                )
                self._write_progress(started=started, completed_steps=completed_steps)

                if completed_steps == next_boundary:
                    if pending:
                        # No request may cross an evaluation boundary.  This
                        # indicates a Controller scheduling regression rather
                        # than a condition to silently race through.
                        raise RuntimeError("Training batches remained in flight at an evaluation boundary.")
                    if self.evaluation_enabled and self.evaluation_interval_steps and (
                        completed_steps % self.evaluation_interval_steps == 0
                    ):
                        adapter_snapshot = self._snapshot_evaluation_adapter(
                            completed_steps=completed_steps,
                            history_item=history_item,
                        )
                        self._run_evaluation(
                            client,
                            completed_steps=completed_steps,
                            actor_manager=actor_manager,
                            learner_manager=learner_manager,
                            teacher_manager=teacher_manager,
                            adapter_snapshot=adapter_snapshot,
                        )
                        self._write_progress(started=started, completed_steps=completed_steps)

            final = {
                "run_name": self.run_name,
                "completed_steps": completed_steps,
                "elapsed_seconds": time.time() - started,
                "output_dir": str(self.output_dir),
                "evaluations": self.evaluation_history,
            }
            write_json(self.output_dir / "run_complete.json", _json_number(final))
            print(f"[Controller] Training completed: {final}", flush=True)
            return final
        except BaseException as error:
            run_error = error
            raise
        finally:
            try:
                self._shutdown_managers(
                    actor_manager=actor_manager,
                    teacher_manager=teacher_manager,
                    learner_manager=learner_manager,
                )
            finally:
                for supervisor in supervisors:
                    supervisor.join(timeout=10.0)
                close = getattr(client, "close", None)
                if callable(close):
                    close()
                self._swanlab.finish(
                    state="crashed" if run_error is not None else "success",
                    error=run_error,
                )


def _configured_plugin_factory(config: Mapping[str, object]) -> PluginFactory:
    """Build the spawn-safe Worker plugin factory selected by configuration.

    Algorithm YAML deliberately names a registered plugin *class*, rather
    than a module-level factory function.  The registry owns construction so
    every Worker receives the same pickle-safe callable while the algorithm
    configuration stays declarative and readable.
    """

    return plugin_factory_from_config(config)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch FrugalOPL's Controller-managed training flow.")
    config_source = parser.add_mutually_exclusive_group(required=True)
    config_source.add_argument(
        "--config",
        help="Path to a complete training YAML whose extends chain composes framework and profile layers.",
    )
    config_source.add_argument(
        "--framework-config",
        help="Compatibility mode: path to a framework YAML used with --algorithm-config.",
    )
    parser.add_argument(
        "--algorithm-config",
        default=None,
        help="Compatibility mode: algorithm overlay YAML used with --framework-config.",
    )
    args = parser.parse_args(argv)
    if args.config:
        if args.algorithm_config is not None:
            parser.error("--algorithm-config may only be used together with --framework-config.")
        config = load_merged_config(args.config)
    else:
        if args.algorithm_config is None:
            parser.error("--framework-config requires --algorithm-config.")
        config = load_merged_config(args.framework_config, args.algorithm_config)
    controller = Controller(config, plugin_factory=_configured_plugin_factory(config))
    controller.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[Controller] Training interrupted.", file=sys.stderr, flush=True)
        raise SystemExit(130) from None


__all__ = ["Controller", "PluginFactory", "main"]
