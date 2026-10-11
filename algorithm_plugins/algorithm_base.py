"""Base contract for FrugalOPL algorithm plugins.

The public plugin surface deliberately contains no Worker, Redis, device, or
distributed-runtime object.  Worker-local helpers are exposed as ordinary
instance methods and resolved only for the duration of the hook that owns
them.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal, Protocol

from utils.algorithm_runtime import AlgorithmRuntimeConfig

Signals = dict[str, object]
Metrics = dict[str, object]


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
    logical batch outside the plugin.  Plugins need implement only the methods
    named by their own pipeline.
    """

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

        Learner calls are available only while the configured objective runs.
        Their supplied ``batch`` contains exactly one rollout: all model
        tensors retain an explicit leading dimension of ``1`` and contain no
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
        if scope.phase != "objective":
            raise RuntimeError("compute_logits() is only available in the configured Learner objective.")
        return scope.runtime._plugin_compute_logits(
            dict(batch),
            mode=mode,
            prompts=prompts,
            prompt_ids=prompt_ids,
            return_signals=dict(return_signals) if return_signals is not None else None,
        )


__all__ = ["AlgorithmPlugin", "Metrics", "Signals"]
