"""Offline, multi-GPU evaluation through FrugalOPL's generic Actor.

This module deliberately does not start Redis, a Controller, a Learner, or a
Teacher.  It launches one short-lived Actor worker per configured GPU, asks it
to generate evaluation trajectories directly, and writes deterministic JSON
results to the run output directory.

The Actor owns prompt rendering: every dataset record is passed through
unchanged, so a template may use ``{question}``, ``{solution}``, ``{answer}``,
or any other same-named field present in that record.  This file therefore
never assumes that ``question`` is the only prompt variable.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# ``python utils/eval.py`` sets sys.path[0] to ``utils/`` rather than the
# repository root.  Make package imports stable even when start_eval.sh was
# launched from another directory.
_MODULE_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_MODULE_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_PROJECT_ROOT))

from utils.answer_verifier import extract_final_answer, verify_response_answer


MAX_EVALUATION_ROLLOUTS_PER_BATCH = 64
_WORKER_READY = "ready"
_WORKER_RESULT = "result"
_WORKER_ERROR = "error"


def _project_root() -> Path:
    """Return the repository root independently of the caller's CWD."""

    return Path(__file__).resolve().parents[1]


def _as_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return dict(value)


def _actor_config_helpers() -> tuple[Any, Any]:
    """Import config helpers without importing vLLM in the parent process."""

    from engine.actor import load_yaml_config, merge_configs

    return load_yaml_config, merge_configs


def _resolve_extended_config_path(raw_path: object, current_config: Path) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("Top-level 'extends' must be a non-empty YAML path or a list of paths.")
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = current_config.parent / candidate
    return candidate.resolve()


def _load_evaluation_config_tree(path: Path, ancestry: tuple[Path, ...] = ()) -> dict[str, object]:
    """Load a YAML config and recursively merge its explicit ``extends`` bases."""

    resolved = path.expanduser().resolve()
    if resolved in ancestry:
        chain = " -> ".join(str(item) for item in (*ancestry, resolved))
        raise ValueError(f"Cyclic evaluation config inheritance: {chain}")

    load_yaml_config, merge_configs = _actor_config_helpers()
    overlay = dict(load_yaml_config(resolved))
    extends = overlay.pop("extends", None)
    if extends is None:
        return overlay

    if isinstance(extends, str):
        base_entries: list[object] = [extends]
    elif isinstance(extends, Sequence) and not isinstance(extends, (bytes, bytearray)):
        base_entries = list(extends)
    else:
        raise ValueError("Top-level 'extends' must be a YAML path or a list of YAML paths.")
    if not base_entries:
        raise ValueError("Top-level 'extends' may not be an empty list.")

    merged_base: dict[str, object] = {}
    for entry in base_entries:
        base_path = _resolve_extended_config_path(entry, resolved)
        base = _load_evaluation_config_tree(base_path, (*ancestry, resolved))
        merged_base = merge_configs(merged_base, base)
    # ``extends`` has intentionally been popped, so it cannot leak into the
    # runtime configuration seen by Actor.
    return merge_configs(merged_base, overlay)


def load_evaluation_config(path: str | Path) -> dict[str, object]:
    """Load a standalone evaluation YAML or an overlay using ``extends``.

    ``extends`` is resolved relative to the YAML file that declares it.  It is
    an evaluation-only composition instruction and is absent from the returned
    runtime mapping.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Evaluation config does not exist: {resolved}")
    return _load_evaluation_config_tree(resolved)


def _resolve_input_path(path_value: object, *, config_path: Path | None) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("Each evaluation dataset needs a non-empty 'path'.")
    raw = Path(path_value).expanduser()
    if raw.is_absolute():
        return raw.resolve()

    # Framework configs conventionally use project-root-relative paths (for
    # example ``data/eval/AIME-2024.json``).  A sibling-relative fallback also
    # makes a standalone config portable outside this repository.
    project_candidate = _project_root() / raw
    if project_candidate.exists() or config_path is None:
        return project_candidate.resolve()
    return (config_path.parent / raw).resolve()


def _read_json_records(path: Path) -> object:
    with path.open("r", encoding="utf-8") as file:
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            return [json.loads(line) for line in file if line.strip()]
        return json.load(file)


def _extract_record_list(payload: object, path: Path) -> list[object]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        # A few common dataset wrappers are inexpensive to support and still
        # keep all original record keys available to prompt templates.
        for field in ("records", "data", "examples"):
            candidate = payload.get(field)
            if isinstance(candidate, list):
                return candidate
    raise ValueError(f"Evaluation dataset must be a JSON list of records: {path}")


def load_dataset_records(
    dataset_config: Mapping[str, object],
    *,
    config_path: str | Path | None = None,
    default_max_examples: object = None,
) -> list[dict[str, object]]:
    """Load records while preserving every dataset field for Actor rendering."""

    cfg = _as_mapping(dataset_config, "evaluation dataset")
    resolved_config = Path(config_path).expanduser().resolve() if config_path is not None else None
    dataset_path = _resolve_input_path(cfg.get("path"), config_path=resolved_config)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Evaluation dataset does not exist: {dataset_path}")

    raw_records = _extract_record_list(_read_json_records(dataset_path), dataset_path)
    max_examples = cfg.get("max_examples", default_max_examples)
    if max_examples is not None:
        if isinstance(max_examples, bool):
            raise ValueError("dataset max_examples must be a positive integer or null.")
        try:
            count = int(max_examples)
        except (TypeError, ValueError) as error:
            raise ValueError("dataset max_examples must be a positive integer or null.") from error
        if count <= 0:
            raise ValueError("dataset max_examples must be a positive integer or null.")
        raw_records = raw_records[:count]

    records: list[dict[str, object]] = []
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"Dataset record {index} in {dataset_path} must be a mapping.")
        record = copy.deepcopy(dict(raw_record))
        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Dataset record {index} in {dataset_path} needs a non-empty string 'question'.")
        if "answer" not in record or record["answer"] is None:
            raise ValueError(f"Dataset record {index} in {dataset_path} needs a non-null 'answer'.")
        records.append(record)

    if not records:
        raise ValueError(f"Evaluation dataset is empty after max_examples filtering: {dataset_path}")
    return records


def evaluation_batch_limits(evaluation_config: Mapping[str, object]) -> tuple[int, int, int]:
    """Return ``(rollouts_per_batch, rollouts_per_prompt, prompts_per_batch)``.

    The hard 64-rollout cap is intentionally enforced here instead of relying
    on vLLM to fail later, because it is a project-wide local evaluation rule.
    """

    cfg = _as_mapping(evaluation_config, "actor.evaluation")
    try:
        rollouts_per_batch = int(cfg.get("rollouts_per_batch", MAX_EVALUATION_ROLLOUTS_PER_BATCH))
        rollouts_per_prompt = int(cfg.get("rollouts_per_prompt", 8))
    except (TypeError, ValueError) as error:
        raise ValueError("Evaluation rollout counts must be integers.") from error
    if not 1 <= rollouts_per_batch <= MAX_EVALUATION_ROLLOUTS_PER_BATCH:
        raise ValueError(
            "actor.evaluation.rollouts_per_batch must be between 1 and "
            f"{MAX_EVALUATION_ROLLOUTS_PER_BATCH}."
        )
    if rollouts_per_prompt < 1:
        raise ValueError("actor.evaluation.rollouts_per_prompt must be positive.")
    if rollouts_per_prompt > rollouts_per_batch:
        raise ValueError(
            "actor.evaluation.rollouts_per_prompt may not exceed "
            "actor.evaluation.rollouts_per_batch."
        )
    return rollouts_per_batch, rollouts_per_prompt, rollouts_per_batch // rollouts_per_prompt


def split_records_into_rollout_batches(
    records: Sequence[Mapping[str, object]],
    *,
    rollouts_per_batch: int,
    rollouts_per_prompt: int,
) -> list[list[dict[str, object]]]:
    """Split records so every Actor call stays within the rollout cap."""

    if rollouts_per_batch < 1 or rollouts_per_prompt < 1:
        raise ValueError("rollout counts must be positive.")
    if rollouts_per_batch > MAX_EVALUATION_ROLLOUTS_PER_BATCH:
        raise ValueError(
            f"rollouts_per_batch may not exceed {MAX_EVALUATION_ROLLOUTS_PER_BATCH}."
        )
    prompts_per_batch = rollouts_per_batch // rollouts_per_prompt
    if prompts_per_batch < 1:
        raise ValueError("rollouts_per_prompt may not exceed rollouts_per_batch.")

    batches: list[list[dict[str, object]]] = []
    for start in range(0, len(records), prompts_per_batch):
        batch: list[dict[str, object]] = []
        for record in records[start : start + prompts_per_batch]:
            if not isinstance(record, Mapping):
                raise TypeError("Evaluation records must be mappings.")
            batch.append(copy.deepcopy(dict(record)))
        if batch:
            batches.append(batch)
    return batches


def _as_list(value: object, field: str) -> list[object]:
    """Convert ordinary sequences and CPU tensors to a plain list."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"trajectory['{field}'] must be a sequence.")
    return list(value)


def _token_length(value: object, response: str) -> int:
    if value is not None:
        numel = getattr(value, "numel", None)
        if callable(numel):
            return int(numel())
        try:
            return len(value)  # type: ignore[arg-type]
        except TypeError:
            pass
    # Actor always returns response_ids.  This fallback makes a small mocked
    # Actor still usable in CPU tests without pretending its count is exact.
    return len(response.split())


def _flat_trajectory_field(
    trajectory: Mapping[str, object],
    field: str,
    expected_count: int,
    *,
    default: object = None,
) -> list[object]:
    if field not in trajectory:
        return [default] * expected_count
    values = _as_list(trajectory[field], field)
    if len(values) != expected_count:
        raise ValueError(
            f"trajectory['{field}'] has {len(values)} entries, expected {expected_count}."
        )
    return values


def build_question_results(
    records: Sequence[Mapping[str, object]],
    trajectory: Mapping[str, object],
    *,
    rollouts_per_prompt: int,
    save_rollouts: bool,
    start_index: int = 0,
) -> list[dict[str, object]]:
    """Attach verifier results to one Actor trajectory without framework state.

    This is intentionally a pure transformation.  It is used by the runtime
    and can be exercised with a small fake trajectory in CPU-only tests.
    """

    if rollouts_per_prompt < 1:
        raise ValueError("rollouts_per_prompt must be positive.")
    if not isinstance(trajectory, Mapping):
        raise TypeError("Actor evaluation result 'traj' must be a mapping.")
    normalized_records = [dict(record) for record in records]
    for index, record in enumerate(normalized_records):
        if "answer" not in record:
            raise ValueError(f"Evaluation record {start_index + index} is missing 'answer'.")

    raw_responses = _as_list(trajectory.get("responses", []), "responses")
    responses = [str(response) for response in raw_responses]
    expected_count = len(responses)
    response_ids = _flat_trajectory_field(trajectory, "response_ids", expected_count)
    truncated = _flat_trajectory_field(
        trajectory, "response_truncated", expected_count, default=False
    )

    if "prompt_indices" in trajectory:
        raw_indices = _flat_trajectory_field(trajectory, "prompt_indices", expected_count)
        try:
            prompt_indices = [int(index) for index in raw_indices]
        except (TypeError, ValueError) as error:
            raise ValueError("trajectory['prompt_indices'] must contain integers.") from error
    else:
        expected = len(normalized_records) * rollouts_per_prompt
        if expected_count != expected:
            raise ValueError(
                "Actor trajectory omitted prompt_indices and does not contain the expected "
                f"number of responses ({expected_count} != {expected})."
            )
        prompt_indices = [index // rollouts_per_prompt for index in range(expected_count)]

    grouped: list[list[dict[str, object]]] = [[] for _ in normalized_records]
    for response_index, (prompt_index, response, token_ids, was_truncated) in enumerate(
        zip(prompt_indices, responses, response_ids, truncated, strict=True)
    ):
        if not 0 <= prompt_index < len(normalized_records):
            raise ValueError(
                f"trajectory['prompt_indices'][{response_index}]={prompt_index} is outside "
                f"the batch of {len(normalized_records)} records."
            )
        record = normalized_records[prompt_index]
        score = float(verify_response_answer(response, str(record["answer"])))
        boxed_answer, format_valid = extract_final_answer(response)
        rollout: dict[str, object] = {
            "rollout_index": len(grouped[prompt_index]),
            "score": score,
            "correct": bool(score),
            "response_token_length": _token_length(token_ids, response),
            "response_truncated": bool(was_truncated),
            "boxed_answer": boxed_answer if format_valid else None,
        }
        if save_rollouts:
            rollout["response"] = response
        grouped[prompt_index].append(rollout)

    question_results: list[dict[str, object]] = []
    for record_index, (record, rollouts) in enumerate(zip(normalized_records, grouped, strict=True)):
        if len(rollouts) != rollouts_per_prompt:
            raise ValueError(
                f"Actor returned {len(rollouts)} responses for record {start_index + record_index}; "
                f"expected rollouts_per_prompt={rollouts_per_prompt}."
            )
        correct_count = sum(bool(rollout["correct"]) for rollout in rollouts)
        question_results.append(
            {
                "record_index": start_index + record_index,
                "question": record.get("question"),
                "answer": record.get("answer"),
                "correct_count": correct_count,
                "incorrect_count": len(rollouts) - correct_count,
                "rollouts": rollouts,
            }
        )
    return question_results


def compute_metrics(
    question_results: Sequence[Mapping[str, object]], *, rollouts_per_prompt: int | None = None
) -> dict[str, object]:
    """Compute length, truncation rate, Pass@1, Avg@N, and Pass@N.

    ``N`` is materialized in the returned metric names (for example,
    ``avg_at_4`` and ``pass_at_4``).  This keeps persisted JSON and external
    telemetry self-describing instead of leaving an ambiguous ``*_at_k`` key
    whose value must be interpreted through a separate field.
    """

    if not question_results:
        raise ValueError("Cannot compute evaluation metrics for zero questions.")
    first_scores: list[float] = []
    per_question_accuracy: list[float] = []
    per_question_any_correct: list[float] = []
    response_lengths: list[int] = []
    total_correct = 0
    total_responses = 0
    total_truncated = 0

    inferred_k: int | None = rollouts_per_prompt
    for question_index, result in enumerate(question_results):
        rollouts_obj = result.get("rollouts") if isinstance(result, Mapping) else None
        if not isinstance(rollouts_obj, Sequence) or isinstance(rollouts_obj, (str, bytes)):
            raise TypeError(f"Question result {question_index} needs a 'rollouts' sequence.")
        rollouts = list(rollouts_obj)
        if not rollouts:
            raise ValueError(f"Question result {question_index} has no rollouts.")
        if inferred_k is None:
            inferred_k = len(rollouts)
        if len(rollouts) != inferred_k:
            raise ValueError("Every question must have the same number of rollouts.")

        scores: list[float] = []
        for rollout_index, rollout in enumerate(rollouts):
            if not isinstance(rollout, Mapping):
                raise TypeError(f"Rollout {rollout_index} for question {question_index} must be a mapping.")
            score = float(rollout.get("score", float(bool(rollout.get("correct", False)))))
            if not math.isfinite(score):
                raise ValueError("Rollout scores must be finite.")
            scores.append(score)
            total_truncated += bool(rollout.get("response_truncated", False))
            try:
                response_lengths.append(int(rollout.get("response_token_length", 0)))
            except (TypeError, ValueError) as error:
                raise ValueError("response_token_length must be an integer.") from error

        first_scores.append(scores[0])
        per_question_accuracy.append(sum(scores) / len(scores))
        per_question_any_correct.append(float(any(score > 0.0 for score in scores)))
        total_correct += sum(score > 0.0 for score in scores)
        total_responses += len(scores)

    assert inferred_k is not None
    avg_at_n_key = f"avg_at_{inferred_k}"
    pass_at_n_key = f"pass_at_{inferred_k}"
    metrics: dict[str, object] = {
        "question_count": len(question_results),
        "rollouts_per_prompt": inferred_k,
        "response_count": total_responses,
        "correct_response_count": total_correct,
        "incorrect_response_count": total_responses - total_correct,
        "truncation_rate": total_truncated / total_responses,
        "average_response_length": sum(response_lengths) / len(response_lengths),
        "pass_at_1": sum(first_scores) / len(first_scores),
    }
    metrics[avg_at_n_key] = sum(per_question_accuracy) / len(per_question_accuracy)
    # For a single rollout, Pass@N is exactly the already explicit Pass@1
    # metric. Avoid assigning the same key twice (which could otherwise mask
    # the Pass@1 score if a caller supplies non-binary rollout scores).
    if pass_at_n_key != "pass_at_1":
        metrics[pass_at_n_key] = sum(per_question_any_correct) / len(per_question_any_correct)
    return metrics


def _call_actor_evaluation_api(
    actor: object,
    records: list[dict[str, object]],
    prompt_alias: str | None,
) -> dict[str, object]:
    """Call the public offline Actor API with a clear compatibility error."""

    for method_name in (
        "generate_evaluation_rollouts",
        "evaluate_records",
        "process_evaluation_records",
    ):
        method = getattr(actor, method_name, None)
        if not callable(method):
            continue
        result = method(records, prompt_alias=prompt_alias)
        if not isinstance(result, Mapping):
            raise TypeError(f"Actor.{method_name}() must return a mapping.")
        if "traj" not in result:
            raise ValueError(f"Actor.{method_name}() result must contain a 'traj' mapping.")
        return dict(result)
    raise RuntimeError(
        "Actor needs a public offline evaluation method: "
        "generate_evaluation_rollouts(records, prompt_alias=...)."
    )


def _evaluation_worker(
    gpu_id: int,
    config: Mapping[str, object],
    task_queue: Any,
    result_queue: Any,
) -> None:
    """Run one directly-addressed Actor without Redis in a spawned process."""

    # This must precede the first Actor/vLLM/torch import in this process.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    actor: object | None = None
    try:
        from engine.actor import Actor

        actor = Actor(config, gpu_id=gpu_id)
        actor.initialize_engine()
        actor.start_generation_loop()
        result_queue.put({"kind": _WORKER_READY, "gpu_id": gpu_id})

        while True:
            task = task_queue.get()
            if task is None:
                break
            if not isinstance(task, Mapping):
                raise TypeError("Evaluation worker received an invalid task.")
            task_id = task.get("task_id")
            records = task.get("records")
            prompt_alias = task.get("prompt_alias")
            if not isinstance(task_id, int):
                raise TypeError("Evaluation worker task_id must be an integer.")
            if not isinstance(records, list) or not all(isinstance(record, Mapping) for record in records):
                raise TypeError("Evaluation worker records must be a list of mappings.")
            if prompt_alias is not None and not isinstance(prompt_alias, str):
                raise TypeError("Evaluation worker prompt_alias must be a string or null.")

            try:
                actor_result = _call_actor_evaluation_api(
                    actor,
                    [dict(record) for record in records],
                    prompt_alias,
                )
                result_queue.put(
                    {
                        "kind": _WORKER_RESULT,
                        "gpu_id": gpu_id,
                        "task_id": task_id,
                        "actor_result": actor_result,
                    }
                )
            except Exception as error:
                result_queue.put(
                    {
                        "kind": _WORKER_ERROR,
                        "gpu_id": gpu_id,
                        "task_id": task_id,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
    except Exception as error:
        result_queue.put(
            {
                "kind": _WORKER_ERROR,
                "gpu_id": gpu_id,
                "task_id": None,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        close = getattr(actor, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def _worker_message_or_raise(result_queue: Any, workers: Sequence[Any]) -> dict[str, object]:
    """Wait for one worker message while detecting a silent child exit."""

    while True:
        try:
            message = result_queue.get(timeout=1.0)
        except queue.Empty:
            exited = [
                f"GPU {getattr(worker, 'name', '?')} (exitcode={getattr(worker, 'exitcode', None)})"
                for worker in workers
                if not worker.is_alive()
            ]
            if exited:
                raise RuntimeError("Evaluation Actor worker exited unexpectedly: " + ", ".join(exited))
            continue
        if not isinstance(message, Mapping):
            raise RuntimeError("Evaluation worker returned an invalid result message.")
        normalized = dict(message)
        if normalized.get("kind") == _WORKER_ERROR:
            gpu_id = normalized.get("gpu_id")
            task_id = normalized.get("task_id")
            error = normalized.get("error", "unknown worker error")
            worker_trace = normalized.get("traceback", "")
            raise RuntimeError(
                f"Evaluation Actor failed on GPU {gpu_id}, task {task_id}: {error}\n{worker_trace}"
            )
        return normalized


def _stop_workers(workers: Sequence[Any], task_queue: Any, *, force: bool) -> None:
    for _worker in workers:
        try:
            task_queue.put_nowait(None)
        except Exception:
            break
    for worker in workers:
        if force and worker.is_alive():
            worker.terminate()
        worker.join(timeout=30.0)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5.0)


def execute_actor_tasks(
    config: Mapping[str, object],
    tasks: Sequence[Mapping[str, object]],
) -> dict[int, dict[str, object]]:
    """Execute prepared evaluation batches across one Actor process per GPU."""

    if not tasks:
        return {}
    actor_cfg = _as_mapping(config.get("actor"), "actor")
    raw_gpu_ids = actor_cfg.get("gpu_ids")
    if not isinstance(raw_gpu_ids, list) or not raw_gpu_ids:
        raise ValueError("actor.gpu_ids must be a non-empty list for offline evaluation.")
    try:
        gpu_ids = [int(gpu_id) for gpu_id in raw_gpu_ids]
    except (TypeError, ValueError) as error:
        raise ValueError("actor.gpu_ids must contain integer GPU ids.") from error
    if any(gpu_id < 0 for gpu_id in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("actor.gpu_ids must contain unique non-negative GPU ids.")

    run_cfg = _as_mapping(config.get("run"), "run")
    runtime_cfg = _as_mapping(run_cfg.get("runtime", {}), "run.runtime")
    method = str(runtime_cfg.get("process_start_method", "spawn"))
    context = mp.get_context(method)
    worker_gpu_ids = gpu_ids[: min(len(gpu_ids), len(tasks))]
    task_queue = context.Queue()
    result_queue = context.Queue()
    workers = [
        context.Process(
            target=_evaluation_worker,
            args=(gpu_id, copy.deepcopy(dict(config)), task_queue, result_queue),
            name=f"FrugalOPL-Eval-Actor-GPU{gpu_id}",
        )
        for gpu_id in worker_gpu_ids
    ]

    for worker in workers:
        worker.start()
    forced_stop = True
    try:
        ready_gpu_ids: set[int] = set()
        while len(ready_gpu_ids) < len(workers):
            message = _worker_message_or_raise(result_queue, workers)
            if message.get("kind") != _WORKER_READY:
                raise RuntimeError(f"Expected Actor worker readiness, received: {message!r}")
            gpu_id = message.get("gpu_id")
            if not isinstance(gpu_id, int):
                raise RuntimeError("Actor readiness message lacks an integer gpu_id.")
            ready_gpu_ids.add(gpu_id)

        for task in tasks:
            task_queue.put(copy.deepcopy(dict(task)))

        completed: dict[int, dict[str, object]] = {}
        while len(completed) < len(tasks):
            message = _worker_message_or_raise(result_queue, workers)
            if message.get("kind") != _WORKER_RESULT:
                raise RuntimeError(f"Expected Actor evaluation result, received: {message!r}")
            task_id = message.get("task_id")
            actor_result = message.get("actor_result")
            if not isinstance(task_id, int) or task_id not in range(len(tasks)):
                raise RuntimeError(f"Actor returned an invalid task id: {task_id!r}")
            if task_id in completed:
                raise RuntimeError(f"Actor returned task {task_id} more than once.")
            if not isinstance(actor_result, Mapping):
                raise RuntimeError(f"Actor result for task {task_id} is not a mapping.")
            completed[task_id] = dict(actor_result)

        forced_stop = False
        return completed
    finally:
        _stop_workers(workers, task_queue, force=forced_stop)
        close = getattr(task_queue, "close", None)
        if callable(close):
            close()
        close = getattr(result_queue, "close", None)
        if callable(close):
            close()


def _output_directory(config: Mapping[str, object], *, config_path: Path | None) -> Path:
    run_cfg = _as_mapping(config.get("run"), "run")
    configured = run_cfg.get("output_dir", run_cfg.get("output_path"))
    if configured is None or (isinstance(configured, str) and not configured.strip()):
        policy_cfg = _as_mapping(run_cfg.get("policy"), "run.policy")
        model = str(policy_cfg.get("model_name_or_path", "model")).rstrip("/\\")
        model_name = Path(model).name or "model"
        configured = str(Path("outputs") / model_name)
    if not isinstance(configured, str):
        raise ValueError("run.output_dir (or run.output_path) must be a string.")
    output = Path(configured).expanduser()
    if output.is_absolute():
        return output.resolve()
    # Output locations, like dataset locations, are normally project-root
    # relative.  Do not create a surprising config-directory outputs folder.
    return (_project_root() / output).resolve()


def _json_value(value: object) -> object:
    """Convert common tensor/path values to safe JSON data without torch imports."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_value(tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, value: object) -> None:
    """Atomically write a human-readable JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(_json_value(value), file, ensure_ascii=False, indent=2, sort_keys=False)
        file.write("\n")
    temporary.replace(path)


def _print_metrics(dataset_name: str, metrics: Mapping[str, object]) -> None:
    k = int(metrics["rollouts_per_prompt"])
    avg_at_n_key = f"avg_at_{k}"
    pass_at_n_key = f"pass_at_{k}"
    print(
        f"[{dataset_name}] 平均长度={float(metrics['average_response_length']):.2f} tokens | "
        f"Pass@1={float(metrics['pass_at_1']):.4f} | "
        f"Avg@{k}={float(metrics[avg_at_n_key]):.4f} | "
        f"Pass@{k}={float(metrics[pass_at_n_key]):.4f}",
        flush=True,
    )


def run_evaluation(
    config: Mapping[str, object],
    *,
    config_path: str | Path | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Run all configured datasets and persist ``summary.json`` / ``details.json``.

    The return value is ``(summary, details)`` so callers and tests do not
    need to re-read output files.  Only Actor workers touch CUDA/vLLM.
    """

    runtime_config = copy.deepcopy(_as_mapping(config, "evaluation config"))
    config_file = Path(config_path).expanduser().resolve() if config_path is not None else None
    run_cfg = _as_mapping(runtime_config.get("run"), "run")
    actor_cfg = _as_mapping(runtime_config.get("actor"), "actor")
    evaluation_cfg = _as_mapping(actor_cfg.get("evaluation"), "actor.evaluation")
    if not bool(evaluation_cfg.get("enabled", True)):
        raise ValueError("actor.evaluation.enabled must be true for offline evaluation.")
    rollouts_per_batch, rollouts_per_prompt, _prompts_per_batch = evaluation_batch_limits(evaluation_cfg)
    save_rollouts = bool(evaluation_cfg.get("save_rollouts", False))
    datasets = _as_mapping(evaluation_cfg.get("datasets"), "actor.evaluation.datasets")
    if not datasets:
        raise ValueError("actor.evaluation.datasets may not be empty.")

    default_max_examples = evaluation_cfg.get("max_examples")
    prepared_datasets: dict[str, dict[str, object]] = {}
    tasks: list[dict[str, object]] = []
    task_id = 0
    for dataset_name, raw_dataset_cfg in datasets.items():
        if not isinstance(dataset_name, str) or not dataset_name:
            raise ValueError("Evaluation dataset names must be non-empty strings.")
        dataset_cfg = _as_mapping(raw_dataset_cfg, f"actor.evaluation.datasets.{dataset_name}")
        prompt_alias = dataset_cfg.get("prompt", "evaluation")
        if not isinstance(prompt_alias, str) or not prompt_alias:
            raise ValueError(f"Evaluation dataset {dataset_name!r} needs a non-empty prompt alias.")
        records = load_dataset_records(
            dataset_cfg,
            config_path=config_file,
            default_max_examples=default_max_examples,
        )
        prepared_datasets[dataset_name] = {
            "records": records,
            "prompt_alias": prompt_alias,
            "question_results": [],
        }
        record_offset = 0
        for batch in split_records_into_rollout_batches(
            records,
            rollouts_per_batch=rollouts_per_batch,
            rollouts_per_prompt=rollouts_per_prompt,
        ):
            tasks.append(
                {
                    "task_id": task_id,
                    "dataset_name": dataset_name,
                    "record_offset": record_offset,
                    "records": batch,
                    "prompt_alias": prompt_alias,
                }
            )
            record_offset += len(batch)
            task_id += 1

    started = time.time()
    actor_results = execute_actor_tasks(runtime_config, tasks)
    elapsed_seconds = time.time() - started
    for task in tasks:
        identifier = int(task["task_id"])
        actor_result = actor_results[identifier]
        trajectory = actor_result.get("traj")
        if not isinstance(trajectory, Mapping):
            raise TypeError(f"Actor result for task {identifier} needs a 'traj' mapping.")
        dataset_name = str(task["dataset_name"])
        details = prepared_datasets[dataset_name]
        question_results = build_question_results(
            task["records"],  # type: ignore[arg-type]
            trajectory,
            rollouts_per_prompt=rollouts_per_prompt,
            save_rollouts=save_rollouts,
            start_index=int(task["record_offset"]),
        )
        stored_results = details["question_results"]
        assert isinstance(stored_results, list)
        stored_results.extend(question_results)

    dataset_summaries: dict[str, object] = {}
    detail_datasets: dict[str, object] = {}
    all_question_results: list[dict[str, object]] = []
    for dataset_name, prepared in prepared_datasets.items():
        question_results = prepared["question_results"]
        assert isinstance(question_results, list)
        question_results.sort(key=lambda result: int(result["record_index"]))
        metrics = compute_metrics(question_results, rollouts_per_prompt=rollouts_per_prompt)
        _print_metrics(dataset_name, metrics)
        dataset_summaries[dataset_name] = {
            "prompt": prepared["prompt_alias"],
            "metrics": metrics,
        }
        detail_datasets[dataset_name] = {
            "prompt": prepared["prompt_alias"],
            "questions": question_results,
        }
        all_question_results.extend(question_results)

    overall_metrics = compute_metrics(all_question_results, rollouts_per_prompt=rollouts_per_prompt)
    if len(dataset_summaries) > 1:
        _print_metrics("overall", overall_metrics)
    output_dir = _output_directory(runtime_config, config_path=config_file)
    run_name = run_cfg.get("name")
    summary: dict[str, object] = {
        "run_name": run_name,
        "output_dir": str(output_dir),
        "elapsed_seconds": elapsed_seconds,
        "overall": overall_metrics,
        "datasets": dataset_summaries,
    }
    details: dict[str, object] = {
        "run_name": run_name,
        "save_rollouts": save_rollouts,
        "datasets": detail_datasets,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "details.json", details)
    print(f"评估结果已写入: {output_dir}", flush=True)
    return summary, details


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run FrugalOPL offline Actor evaluation.")
    parser.add_argument("--config", required=True, help="Standalone YAML or an evaluation overlay YAML.")
    args = parser.parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    config = load_evaluation_config(config_path)
    run_evaluation(config, config_path=config_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("评估已中断。", file=sys.stderr, flush=True)
        raise SystemExit(130) from None


__all__ = [
    "MAX_EVALUATION_ROLLOUTS_PER_BATCH",
    "build_question_results",
    "compute_metrics",
    "evaluation_batch_limits",
    "execute_actor_tasks",
    "load_dataset_records",
    "load_evaluation_config",
    "run_evaluation",
    "split_records_into_rollout_batches",
    "write_json",
]
