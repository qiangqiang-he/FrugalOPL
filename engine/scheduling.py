"""Algorithm-independent global-batch scheduling and placement validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


def _gpu_ids(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list of distinct non-negative GPU integers.")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise ValueError(f"{field} must contain non-negative GPU integers.")
    if len(set(value)) != len(value):
        raise ValueError(f"{field} must not contain duplicate GPU IDs.")
    return tuple(value)


@dataclass(frozen=True)
class SchedulingPlan:
    actor_gpu_ids: tuple[int, ...]
    learner_gpu_ids: tuple[int, ...]
    max_inflight_batches: int
    shared: bool
    overlap: bool = False

    @property
    def actor_batches_per_wave(self) -> int:
        return self.max_inflight_batches if self.shared else max(1, self.max_inflight_batches // 2)

    @property
    def learner_batches_per_turn(self) -> int:
        return self.actor_batches_per_wave

    @property
    def mode(self) -> str:
        if self.overlap:
            return "shared_overlap"
        if not self.shared:
            return "async"
        return "shared_serial" if self.max_inflight_batches == 1 else "shared_batched"


def resolve_scheduling(config: Mapping[str, object]) -> SchedulingPlan:
    """Derive all Actor/Learner widths from the single in-flight batch limit.

    A batch is global: Actor GPUs receive question shards, not independent
    complete batches. Placement permits disjoint or identical GPU sets only.
    """

    actor = config.get("actor", {})
    learner = config.get("learner", {})
    controller = config.get("controller", {})
    if not all(isinstance(section, Mapping) for section in (actor, learner, controller)):
        raise ValueError("actor, learner and controller must be mappings.")
    actor_ids = _gpu_ids(actor.get("gpu_ids"), "actor.gpu_ids")
    learner_ids = _gpu_ids(learner.get("gpu_ids"), "learner.gpu_ids")
    overlap = set(actor_ids) & set(learner_ids)
    if overlap and set(actor_ids) != set(learner_ids):
        raise ValueError("Actor and Learner GPU sets must be disjoint or identical; partial overlap is unsupported.")
    inflight = controller.get("max_inflight_batches", 1)
    if isinstance(inflight, bool) or not isinstance(inflight, int) or inflight < 1:
        raise ValueError("controller.max_inflight_batches must be a positive integer.")
    for worker, section, name in (("actor", actor, "engine"), ("learner", learner, "execution")):
        options = section.get(name, {})
        if not isinstance(options, Mapping):
            raise ValueError(f"{worker}.{name} must be a mapping.")
        if "max_batches_per_turn" in options:
            raise ValueError(
                f"{worker}.{name}.max_batches_per_turn is removed; use controller.max_inflight_batches."
            )
    streaming = controller.get("overlap_actor_learner", False)
    if not isinstance(streaming, bool):
        raise ValueError("controller.overlap_actor_learner must be boolean.")
    if streaming and (not overlap or inflight != 1):
        raise ValueError("Actor/Learner overlap requires identical GPU sets and max_inflight_batches=1.")
    if streaming:
        nodes = config.get("pipeline", [])
        if [node.get("class") for node in nodes] != ["Controller", "Actor", "Learner"]:
            raise ValueError("Actor/Learner overlap requires a Controller -> Actor -> Learner pipeline.")
        plugin = config.get("algorithm", {}).get("plugin")
        supported = {"KLOPSDPlugin", "SelfShotOPSDPlugin", "RatioOPSDPlugin", "TCAPlugin",
                     "FKLPlugin", "FKLTrueAnswerPlugin", "FKLMismatchedAnswerPlugin", "ClipOPSDPlugin", "TRAPlugin",
                     "TRASelfShotPlugin", "TRARandomShotPlugin", "TRATrueAnswerPlugin", "TRAMismatchedAnswerPlugin",
                     "TRAMatchedAnswerPlugin",
                     "TCPOV1Plugin", "TCDOPlugin", "RewardGuidedOPSDPlugin", "GRPOPlugin"}
        if plugin not in supported:
            raise ValueError(f"Actor/Learner overlap has no group-streaming contract for {plugin!r}.")
        ds_config = learner.get("deepspeed", {})
        if (ds_config.get("zero_optimization", {}).get("stage", 0) != 0
                or not ds_config.get("managed_gradient_accumulation", True)
                or ds_config.get("gradient_accumulation_steps", 1) != 1):
            raise ValueError("Actor/Learner overlap requires ZeRO stage 0, managed accumulation and GAS=1.")
    return SchedulingPlan(actor_ids, learner_ids, inflight, bool(overlap), streaming)


def question_shards(question_count: int, gpu_ids: tuple[int, ...]) -> list[tuple[int, int, int]]:
    """Return GPU/start/stop slices without splitting a question's rollout group."""

    if question_count < 1 or not gpu_ids:
        raise ValueError("A global batch requires questions and at least one Actor GPU.")
    active = min(question_count, len(gpu_ids))
    quotient, remainder = divmod(question_count, active)
    result = []
    start = 0
    for index, gpu_id in enumerate(gpu_ids[:active]):
        stop = start + quotient + int(index < remainder)
        result.append((gpu_id, start, stop))
        start = stop
    return result


__all__ = ["SchedulingPlan", "question_shards", "resolve_scheduling"]
