"""Base contract for FrugalOPL algorithm plugins.

The public plugin surface deliberately contains no Worker, Redis, device, or
distributed-runtime object.  Worker-local helpers are exposed as ordinary
instance methods and resolved only for the duration of the hook that owns
them.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    import torch

from utils.algorithm_runtime import AlgorithmRuntimeConfig

Signals = dict[str, object]
Metrics = dict[str, object]


def _validate_forward_kl_request(
    batch: Mapping[str, object],
    teacher_prompts: Sequence[str],
    temperature: float,
    token_clip: float | None,
    chunk_size: int | None,
) -> tuple[list[str], float, float | None, int | None]:
    """Validate the shared public/Worker boundary for one rollout's KL."""

    if not isinstance(batch, Mapping):
        raise TypeError("compute_forward_kl(batch, ...) requires a signal mapping.")
    if isinstance(teacher_prompts, (str, bytes)) or not isinstance(teacher_prompts, Sequence):
        raise TypeError("teacher_prompts must be a sequence containing one prompt string.")
    if len(teacher_prompts) != 1 or not isinstance(teacher_prompts[0], str) or not teacher_prompts[0].strip():
        raise ValueError("compute_forward_kl requires exactly one non-empty Teacher prompt.")
    normalized = []
    for value, name in ((temperature, "temperature"), (token_clip, "token_clip")):
        if name == "token_clip" and value is None:
            normalized.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"compute_forward_kl {name} must be a positive finite number.")
        try:
            value = float(value)
        except OverflowError as error:
            raise ValueError(f"compute_forward_kl {name} must be a positive finite number.") from error
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"compute_forward_kl {name} must be a positive finite number.")
        normalized.append(value)
    if chunk_size is not None and (
        isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0
    ):
        raise ValueError("compute_forward_kl chunk_size must be None or a positive integer.")
    return list(teacher_prompts), normalized[0], normalized[1], chunk_size


@dataclass(frozen=True)
class ObjectiveResult:
    """One micro-batch objective and its explicit token normalization weight.

    ``objective`` is the maximized mean over the final valid token mask.
    Multiplying it by ``valid_token_count`` must recover the sum of the same
    token objective, including every regularization term.  The count must be
    a detached scalar integer tensor or a Python integer, obtained from the
    final mask used by the objective rather than the response's padded or
    unfiltered length.  A zero count is permitted for an empty valid mask.

    Learner uses this weight for ``algorithm.loss_aggregation: token``.
    Sequence aggregation continues to average micro-batch objectives equally.
    Two-value iteration preserves existing ``objective, metrics = result``
    callers without exposing normalization as an experiment metric.

    ``requires_optimizer_step=False`` marks an exactly zero contribution.
    Its tokens still enter the loss denominator, and Learner supplies the
    zero parameter backward needed to match distributed collective slots.
    """

    objective: torch.Tensor
    metrics: Metrics
    valid_token_count: torch.Tensor | int
    token_weighted_metrics: frozenset[str] = frozenset()
    requires_optimizer_step: bool = True

    def __iter__(self) -> Iterator[object]:
        yield self.objective
        yield self.metrics


class _PluginRuntime(Protocol):
    """The private worker capability surface used by :class:`AlgorithmPlugin`.

    This protocol intentionally stays private: algorithm authors only call
    ``self.answer_probe`` and ``self.compute_logits`` below.  It exists so the
    base class can remain importable without importing Transformers, DeepSpeed,
    or a Worker implementation.
    """

    def _plugin_answer_probe(
        self,
        batch: Signals,
        *,
        probability_difference_threshold: float,
    ) -> Signals: ...

    def _plugin_compute_logits(
        self,
        batch: Signals,
        *,
        mode: str | None,
        prompts: Sequence[str] | None,
        prompt_ids: object | None,
        return_signals: Mapping[str, object] | None,
    ) -> Signals: ...

    def _plugin_collect_logits(
        self,
        batch: Signals,
        *,
        mode: str,
        prompts: Sequence[str] | None,
        prompt_ids: object | None,
        return_signals: Mapping[str, object] | None,
    ) -> Signals: ...

    def _plugin_compute_forward_kl(
        self,
        batch: Signals,
        *,
        teacher_prompts: Sequence[str],
        temperature: float = 1.0,
        token_clip: float | None = None,
        chunk_size: int | None = None,
    ) -> Signals: ...

    def _plugin_compute_ratio_kl(
        self, batch: Signals, *, teacher_prompts: Sequence[str], temperature: float,
        plausible_fraction: float, chunk_size: int | None,
    ) -> Signals: ...

    def _plugin_compute_tcdo(self, batch: Signals, *, teacher_prompts: Sequence[str], state: int,
                             temperature: float, plausible_fraction: float, chunk_size: int | None) -> Signals: ...

    def _plugin_compute_tail_suppression(
        self, batch: Signals, *, top_p: float, temperature: float, chunk_size: int | None,
    ) -> Signals: ...

    def _plugin_compute_ocsd(
        self, batch: Signals, *, gamma_pos: float, gamma_neg: float, tau: float,
    ) -> Signals: ...


@dataclass(frozen=True)
class _PluginScope:
    """One active Worker hook invocation in the current worker context."""

    runtime: _PluginRuntime
    worker: Literal["learner", "teacher"]
    phase: Literal["collect", "objective"]


_ACTIVE_PLUGIN_SCOPE: ContextVar[_PluginScope | None] = ContextVar(
    "frugalopl_active_plugin_scope",
    default=None,
)


@contextmanager
def _learner_plugin_scope(
    runtime: _PluginRuntime,
    *,
    phase: Literal["collect", "objective"],
) -> Iterator[None]:
    """Make Learner capabilities available to a plugin for one hook call."""

    token = _ACTIVE_PLUGIN_SCOPE.set(_PluginScope(runtime=runtime, worker="learner", phase=phase))
    try:
        yield
    finally:
        _ACTIVE_PLUGIN_SCOPE.reset(token)


@contextmanager
def _teacher_plugin_scope(runtime: _PluginRuntime) -> Iterator[None]:
    """Expose the Teacher's single no-grad logit capability for one hook."""

    token = _ACTIVE_PLUGIN_SCOPE.set(_PluginScope(runtime=runtime, worker="teacher", phase="collect"))
    try:
        yield
    finally:
        _ACTIVE_PLUGIN_SCOPE.reset(token)


class AlgorithmPlugin:
    """Define worker-local signal collection and the training objective.

    Pipeline configuration binds each Worker to the plugin method it should
    call.  A signal hook receives all upstream detached signals and returns
    only the fields newly produced at that Worker; the engine merges and
    forwards them.  The configured Learner objective runs on one rollout, so
    its physical batch dimension is fixed at one while Learner accumulates the
    logical batch outside the plugin.  A legacy ``(objective, metrics)`` return
    supports sequence aggregation; token aggregation requires ObjectiveResult
    with the count from that objective's final valid mask.  Plugins need
    implement only the methods named by their own pipeline.
    """

    requires_frozen_reference: bool = False
    computes_actor_question_accuracy: bool = False

    @classmethod
    def from_runtime_config(
        cls,
        tokenizer: object,
        runtime: AlgorithmRuntimeConfig,
    ) -> "AlgorithmPlugin":
        """Build this plugin from framework-resolved runtime configuration.

        The framework resolves the merged configuration before Worker startup
        and passes only the scoped runtime inputs to the selected plugin.
        Plugins therefore never traverse framework configuration such as
        ``run`` while they construct their algorithm logic.
        """

        del tokenizer, runtime
        raise NotImplementedError(
            f"Registered algorithm plugin {cls.__name__} must implement "
            "from_runtime_config(tokenizer, runtime)."
        )

    def on_training_step_completed(self, *, step: int, metrics: Mapping[str, object]) -> Metrics:
        """Observe globally reduced metrics after one logical optimizer step.

        All Learner ranks call this hook before publishing the matching
        adapter/checkpoint. Returned scalar diagnostics join that step's
        metrics, while any algorithm state change affects the next step.
        """

        del step, metrics
        return {}

    def prepare_learner_batch(self, signals: Signals) -> Signals:
        """Select training rollouts from the complete logical Actor batch."""

        return signals

    def finalize_actor_batch(self, signals: Signals) -> Signals:
        """Postprocess this Actor shard's generated trajectories."""

        return signals

    def training_state_dict(self) -> dict[str, object]:
        """Return algorithm-owned state to save with the optimizer checkpoint."""

        return {}

    def load_training_state_dict(self, state: Mapping[str, object], *, step: int) -> None:
        """Restore algorithm state after the checkpoint's optimizer is loaded."""

        del step
        if not isinstance(state, Mapping) or state:
            raise ValueError(f"{type(self).__name__} does not support non-empty saved algorithm state.")

    def answer_probe(
        self,
        batch: Signals,
        *,
        probability_difference_threshold: float,
    ) -> Signals:
        """Return reward and start/end answer-probe probabilities for ``batch``.

        This helper is available only while the Learner executes its configured
        signal hook.  The algorithm always passes one
        normal batch; the Learner keeps the rollout group intact, shards probe
        work across local GPUs when applicable, and restores the original
        rollout order before returning.
        """

        if not isinstance(batch, Mapping):
            raise TypeError("answer_probe(batch, ...) requires a signal mapping.")
        threshold = float(probability_difference_threshold)
        scope = _ACTIVE_PLUGIN_SCOPE.get()
        if scope is None:
            raise RuntimeError("answer_probe() is available only inside a Learner plugin hook.")
        if scope.worker != "learner" or scope.phase != "collect":
            raise RuntimeError("answer_probe() is only available in the configured Learner signal hook.")
        return scope.runtime._plugin_answer_probe(
            dict(batch),
            probability_difference_threshold=threshold,
        )

    def compute_logits(
        self,
        batch: Signals,
        *,
        mode: Literal["reference", "policy_no_grad", "policy_grad"] | None = None,
        prompts: Sequence[str] | None = None,
        prompt_ids: object | None = None,
        return_signals: Mapping[str, object] | None = None,
    ) -> Signals:
        """Return token-aligned model signals from the current Worker hook.

        In the Learner, ``mode`` selects the only three supported computations:

        - ``"reference"``: LoRA disabled and no gradient;
        - ``"policy_no_grad"``: current LoRA policy under no-grad;
        - ``"policy_grad"``: current LoRA policy with a retained graph.

        In a configured Teacher signal hook, call this helper without
        ``mode``.  It then uses the Teacher's sole no-grad forward and returns
        detached, rollout-aligned signals for the complete logical batch.

        In the Learner signal hook, only ``reference`` and ``policy_no_grad``
        are allowed. They score the complete logical batch and return ragged
        lists of detached CPU tensors in rollout order; prompt overrides have
        one entry per rollout. The engine owns sharding and single-rollout
        forwards, so the plugin does not split the batch itself.

        During the configured objective, ``batch`` contains exactly one
        rollout: all model tensors retain a leading dimension of ``1`` and contain no
        cross-rollout padding.  Prompt and response masks remain present for
        a uniform signal interface, and are all ``True``. Plugins never
        manage batch size, device, gradient accumulation, or distributed
        placement. ``prompts`` and ``prompt_ids`` are optional, mutually
        exclusive prompt overrides for algorithms such as OPSD; a Learner
        override represents this one rollout only.

        """

        if not isinstance(batch, Mapping):
            raise TypeError("compute_logits(batch, ...) requires a signal mapping.")
        if prompts is not None:
            if isinstance(prompts, (str, bytes)) or not isinstance(prompts, Sequence):
                raise TypeError("prompts must be a sequence of rendered prompt strings.")
            if not all(isinstance(prompt, str) for prompt in prompts):
                raise TypeError("prompts must contain only strings.")
        if prompts is not None and prompt_ids is not None:
            raise ValueError("Use either prompts or prompt_ids, not both.")
        if return_signals is not None and not isinstance(return_signals, Mapping):
            raise TypeError("return_signals must be a mapping when provided.")

        scope = _ACTIVE_PLUGIN_SCOPE.get()
        if scope is None:
            raise RuntimeError("compute_logits() is available only inside a Learner plugin hook.")
        if scope.worker == "teacher":
            if scope.phase != "collect":  # Defensive: the Teacher has one public hook.
                raise RuntimeError("Teacher compute_logits() is only available in the configured Teacher signal hook.")
            if mode is not None:
                raise ValueError("Teacher compute_logits() has no mode; call it without mode=.")
            return scope.runtime._plugin_compute_logits(
                dict(batch),
                mode=None,
                prompts=prompts,
                prompt_ids=prompt_ids,
                return_signals=dict(return_signals) if return_signals is not None else None,
            )
        if mode not in {"reference", "policy_no_grad", "policy_grad"}:
            raise ValueError(
                "mode must be 'reference', 'policy_no_grad', or 'policy_grad'; "
                f"got {mode!r}."
            )
        if scope.phase == "collect":
            if mode == "policy_grad":
                raise RuntimeError("policy_grad compute_logits() is only available in the configured Learner objective.")
            return scope.runtime._plugin_collect_logits(
                dict(batch),
                mode=mode,
                prompts=prompts,
                prompt_ids=prompt_ids,
                return_signals=dict(return_signals) if return_signals is not None else None,
            )
        return scope.runtime._plugin_compute_logits(
            dict(batch),
            mode=mode,
            prompts=prompts,
            prompt_ids=prompt_ids,
            return_signals=dict(return_signals) if return_signals is not None else None,
        )

    def compute_forward_kl(
        self,
        batch: Signals,
        *,
        teacher_prompts: Sequence[str],
        temperature: float = 1.0,
        token_clip: float | None = None,
        chunk_size: int | None = None,
    ) -> Signals:
        """Return exact full-vocabulary Teacher-to-Student KL at response positions.

        This capability is available only during a Learner objective for one
        unpadded rollout. Student inputs come from ``batch``; the sole Teacher
        prompt replaces only its prompt, preserving identical response IDs.
        Teacher scoring uses the configured LoRA version under no-grad and
        eval mode. The default interval (-1) uses the frozen base; Student
        scoring remains differentiable with its own LoRA active.

        Both distributions use ``temperature`` and the complete vocabulary;
        no sampled-token or top-k approximation and no T-squared multiplier is
        applied. Optional ``token_clip`` caps each vocabulary contribution
        ``q * (log(q) - log(p))`` from above before summing vocabulary entries.
        Negative contributions are retained. ``chunk_size`` bounds the number
        of response positions materialized at once and defaults to the Worker
        forward chunk size. Full logits/hidden states never leave the Worker.

        Results retain shape ``[1, response_length]``:
        ``response_forward_kl`` is differentiable and includes the pointwise
        cap; ``response_forward_kl_unclipped`` and
        ``response_forward_kl_clipped_fraction`` are detached diagnostics. The
        latter is the fraction of vocabulary contributions exceeding the cap,
        or zero when the cap is disabled. The plugin owns masked reduction
        into its maximized objective and corresponding valid-token count.
        """

        scope = _ACTIVE_PLUGIN_SCOPE.get()
        if scope is None or scope.worker != "learner" or scope.phase != "objective":
            raise RuntimeError("compute_forward_kl() is only available inside a Learner objective.")
        prompts, temperature, token_clip, chunk_size = _validate_forward_kl_request(
            batch, teacher_prompts, temperature, token_clip, chunk_size,
        )
        return scope.runtime._plugin_compute_forward_kl(
            dict(batch), teacher_prompts=prompts, temperature=temperature,
            token_clip=token_clip, chunk_size=chunk_size,
        )

    def compute_ratio_kl(
        self, batch: Signals, *, teacher_prompts: Sequence[str],
        temperature: float = 1.0, plausible_fraction: float = 0.1,
        chunk_size: int | None = None,
    ) -> Signals:
        """Score KL(stopgrad(normalize(p²/q * 1[p>=fraction*pmax])) || p).

        Reuse bounded, checkpointed full-vocabulary paired scoring. Targets
        and the privileged Teacher are detached; only Student LoRA learns.
        """
        scope = _ACTIVE_PLUGIN_SCOPE.get()
        if scope is None or scope.worker != "learner" or scope.phase != "objective":
            raise RuntimeError("compute_ratio_kl() is only available inside a Learner objective.")
        prompts, temperature, _, chunk_size = _validate_forward_kl_request(
            batch, teacher_prompts, temperature, None, chunk_size,
        )
        if (isinstance(plausible_fraction, bool) or not isinstance(plausible_fraction, (int, float))
                or not math.isfinite(plausible_fraction) or not 0 < plausible_fraction <= 1):
            raise ValueError("plausible_fraction must lie in (0, 1].")
        return scope.runtime._plugin_compute_ratio_kl(
            dict(batch), teacher_prompts=prompts, temperature=temperature,
            plausible_fraction=float(plausible_fraction), chunk_size=chunk_size,
        )

    def compute_tcdo(self, batch: Signals, *, teacher_prompts: Sequence[str], state: int,
                     temperature: float = 1.0, plausible_fraction: float = 0.01,
                     chunk_size: int | None = None) -> Signals:
        scope = _ACTIVE_PLUGIN_SCOPE.get()
        if scope is None or scope.worker != "learner" or scope.phase != "objective":
            raise RuntimeError("compute_tcdo requires a Learner objective scope")
        prompts, temperature, _, chunk_size = _validate_forward_kl_request(batch, teacher_prompts, temperature, None, chunk_size)
        if state not in (1, 2) or not 0 < plausible_fraction <= 1:
            raise ValueError("Invalid TCDO state or support threshold")
        return scope.runtime._plugin_compute_tcdo(dict(batch), teacher_prompts=prompts, state=state,
            temperature=temperature, plausible_fraction=plausible_fraction, chunk_size=chunk_size)

    def compute_tail_suppression(
        self, batch: Signals, *, top_p: float = 0.99, temperature: float = 1.0,
        chunk_size: int | None = None,
    ) -> Signals:
        """Score exact KL against the frozen rollout policy's top-p target.

        Available only in a one-rollout Learner objective. The runtime uses
        batch.policy_version, identical prompt/response IDs, and a detached
        old-policy adapter. Vocabulary tensors stay in checkpointed chunks
        on the Worker. Returns token KL, the active-position mask and detached
        diagnostics; plugins own masked reduction and valid_token_count.
        """

        scope = _ACTIVE_PLUGIN_SCOPE.get()
        if scope is None or scope.worker != "learner" or scope.phase != "objective":
            raise RuntimeError("compute_tail_suppression() is only available inside a Learner objective.")
        if not isinstance(batch, Mapping):
            raise TypeError("compute_tail_suppression requires a signal mapping.")
        return scope.runtime._plugin_compute_tail_suppression(
            dict(batch), top_p=top_p, temperature=temperature, chunk_size=chunk_size,
        )

    def compute_ocsd(self, batch: Signals, *, gamma_pos: float, gamma_neg: float, tau: float) -> Signals:
        """Score full-vocabulary OCSD using the current student at temperature one."""
        scope = _ACTIVE_PLUGIN_SCOPE.get()
        if scope is None or scope.worker != "learner" or scope.phase != "objective":
            raise RuntimeError("compute_ocsd() is only available inside a Learner objective.")
        if not isinstance(batch, Mapping):
            raise TypeError("compute_ocsd requires a signal mapping.")
        return scope.runtime._plugin_compute_ocsd(dict(batch), gamma_pos=gamma_pos, gamma_neg=gamma_neg, tau=tau)


__all__ = ["AlgorithmPlugin", "Metrics", "ObjectiveResult", "Signals"]
