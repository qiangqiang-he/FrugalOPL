"""Resolve the framework-owned portion of an algorithm's runtime settings.

Algorithm plugins should not traverse the merged training configuration just
to find framework-wide inputs such as the training prompt, sampling settings,
or Qwen3's thinking mode.  This module owns that small boundary: the
Controller resolves the complete merged configuration once, pre-binds the
result to a Worker-local plugin factory, and plugins receive the resulting
:class:`AlgorithmRuntimeConfig` instead.

The algorithm mapping is deliberately retained as a separate snapshot.  It
contains plugin-owned settings (for example ``advantage_clip``) as well as
the class/name metadata that the registry consumes.  The utility validates
only the shared keys; each plugin remains responsible for validating its own
settings.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, cast

from utils import prompts as prompt_templates


class ChatTemplateTokenizer(Protocol):
    """Minimal tokenizer surface shared by Actor-side algorithm plugins.

    Some supported tokenizer versions do not yet accept ``enable_thinking``.
    Call :func:`apply_generation_chat_template` rather than invoking this
    protocol method directly so that compatibility behaviour stays uniform.
    """

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: object,
    ) -> str: ...


@dataclass(frozen=True)
class AlgorithmRuntimeConfig:
    """Shared, already-resolved inputs used to construct an algorithm plugin.

    Attributes:
        train_template: The actual exported string template named by
            ``run.prompts.train``.  Plugins therefore do not need to inspect
            ``run`` or resolve prompt-module attributes themselves.  It is
            ``None`` when the algorithm does not request a training prompt.
        training_sampling: A private snapshot of
            ``algorithm.training_sampling``.  The Actor receives this
            algorithm-owned mapping unchanged when a plugin requests a
            rollout.  It is ``None`` for algorithms without Actor sampling.
        enable_thinking: The one training/evaluation chat-template mode owned
            by the algorithm configuration.
        algorithm: A private snapshot of the whole ``algorithm`` mapping.
            Plugins read their scoped fields from here, for example
            ``runtime.algorithm[\"advantage_clip\"]``.  Common fields above are
            exposed separately so plugin code does not repeatedly parse them.

    The snapshots are intentionally independent of the source merged config:
    later config mutation cannot silently change a Worker-local plugin.
    """

    train_template: str | None
    training_sampling: Mapping[str, object] | None
    enable_thinking: bool
    algorithm: Mapping[str, object]


def resolve_algorithm_runtime_config(config: Mapping[str, object]) -> AlgorithmRuntimeConfig:
    """Extract shared algorithm runtime inputs from one merged configuration.

    This function is deliberately independent of any plugin class, Worker,
    tokenizer, Torch, or CUDA runtime.  It validates only the framework-owned
    configuration contract:

    * an explicitly configured ``run.prompts.train`` names an exported string
      from :mod:`utils.prompts`;
    * an explicitly configured ``algorithm.training_sampling`` is a mapping;
    * ``algorithm.enable_thinking`` is a boolean when present (and defaults to
      ``False`` for direct/minimal configurations).

    Plugin-specific fields remain untouched in ``AlgorithmRuntimeConfig.algorithm``.
    Neither a prompt nor rollout sampling is universal: a future algorithm can
    omit either field and call the corresponding ``require_*`` helper only if
    it actually needs that capability.
    """

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping.")

    run = config.get("run")
    if run is None:
        prompts: Mapping[str, object] | None = None
    elif not isinstance(run, Mapping):
        raise ValueError("config must contain a mapping at 'run'.")
    else:
        configured_prompts = run.get("prompts")
        if configured_prompts is None:
            prompts = None
        elif not isinstance(configured_prompts, Mapping):
            raise ValueError("config must contain a mapping at 'run.prompts'.")
        else:
            prompts = configured_prompts

    if prompts is None:
        train_template: str | None = None
    else:
        template_name = prompts.get("train")
        if template_name is None:
            train_template = None
        elif not isinstance(template_name, str) or not template_name.strip():
            raise ValueError("run.prompts.train must name an exported prompt template.")
        else:
            template_name = template_name.strip()
            resolved_template = getattr(prompt_templates, template_name, None)
            if template_name not in prompt_templates.__all__ or not isinstance(resolved_template, str):
                raise ValueError(
                    "run.prompts.train must name a string template exported by utils.prompts; "
                    f"got {template_name!r}."
                )
            train_template = resolved_template

    algorithm = config.get("algorithm")
    if not isinstance(algorithm, Mapping):
        raise ValueError("config must contain an algorithm mapping.")
    configured_sampling = algorithm.get("training_sampling")
    if configured_sampling is None:
        training_sampling: Mapping[str, object] | None = None
    elif not isinstance(configured_sampling, Mapping):
        raise ValueError("algorithm.training_sampling must be a mapping.")
    else:
        training_sampling = copy.deepcopy(dict(configured_sampling))
    enable_thinking = algorithm.get("enable_thinking", False)
    if not isinstance(enable_thinking, bool):
        raise TypeError("algorithm.enable_thinking must be a boolean.")

    # The algorithm snapshot lets a plugin read its own fields without seeing
    # framework siblings such as ``run``.  Sampling receives its own copy so a
    # local Actor request can never mutate this scoped snapshot either.
    return AlgorithmRuntimeConfig(
        train_template=train_template,
        training_sampling=training_sampling,
        enable_thinking=enable_thinking,
        algorithm=copy.deepcopy(dict(algorithm)),
    )


def require_train_template(runtime: AlgorithmRuntimeConfig) -> str:
    """Return a training template for an algorithm that renders Actor prompts.

    A prompt is not a universal algorithm requirement: reference-only or
    purely learner-side methods can omit it.  Plugins that do render training
    prompts call this helper during construction and receive a precise config
    error if it was not supplied.
    """

    _require_runtime_config(runtime)
    if runtime.train_template is None:
        raise ValueError("algorithm requires run.prompts.train to name an exported prompt template.")
    return runtime.train_template


def require_training_sampling(runtime: AlgorithmRuntimeConfig) -> Mapping[str, object]:
    """Return rollout sampling settings for an algorithm that uses an Actor.

    The returned mapping is an independent copy so plugin-side signal request
    construction cannot mutate the pre-bound Worker runtime config.
    """

    _require_runtime_config(runtime)
    if runtime.training_sampling is None:
        raise ValueError("algorithm requires algorithm.training_sampling to be a mapping.")
    return copy.deepcopy(dict(runtime.training_sampling))


def _require_runtime_config(runtime: object) -> None:
    if not isinstance(runtime, AlgorithmRuntimeConfig):
        raise TypeError("runtime must be an AlgorithmRuntimeConfig.")


def require_chat_template_tokenizer(tokenizer: object) -> ChatTemplateTokenizer:
    """Validate and return the tokenizer capability algorithm plugins need.

    This keeps a malformed tokenizer from failing later in an unrelated
    signal hook.  It intentionally does not require a concrete Transformers
    tokenizer type, which keeps plugins easy to unit test and compatible with
    supported tokenizer implementations.
    """

    if not callable(getattr(tokenizer, "apply_chat_template", None)):
        raise TypeError("tokenizer must provide apply_chat_template().")
    return cast(ChatTemplateTokenizer, tokenizer)


def apply_generation_chat_template(
    tokenizer: object,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool,
) -> str:
    """Render one generation-ready chat prompt with a narrow compatibility fallback.

    Qwen3 tokenizers accept ``enable_thinking``; older tokenizer versions and
    lightweight test doubles may not.  Retry without that keyword only for
    the exact unsupported-keyword case, leaving all other rendering errors
    visible to the caller.
    """

    if not isinstance(enable_thinking, bool):
        raise TypeError("enable_thinking must be a boolean.")
    apply_template: Callable[..., object] = require_chat_template_tokenizer(tokenizer).apply_chat_template
    try:
        rendered = apply_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError as error:
        message = str(error).lower()
        if "enable_thinking" not in message or "keyword" not in message:
            raise
        rendered = apply_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    if not isinstance(rendered, str):
        raise TypeError("tokenizer.apply_chat_template(..., tokenize=False) must return a string.")
    return rendered


def controller_questions_and_answers(signals: Mapping[str, object]) -> tuple[list[str], list[object]]:
    """Validate the Controller's question/answer batch without changing it.

    Actor-side plugins all receive this same framework signal.  Keeping the
    check here prevents each algorithm from reimplementing it with subtly
    different acceptance rules.
    """

    if not isinstance(signals, Mapping):
        raise TypeError("signals must be a mapping.")
    questions = signals.get("questions")
    answers = signals.get("answers")
    if not isinstance(questions, list) or not all(isinstance(question, str) for question in questions):
        raise TypeError("signals['questions'] must be a list[str].")
    if not isinstance(answers, list):
        raise TypeError("signals['answers'] must be a list.")
    if len(answers) != len(questions):
        raise ValueError("signals['questions'] and signals['answers'] must have the same length and be aligned.")
    return questions, answers


__all__ = [
    "AlgorithmRuntimeConfig",
    "ChatTemplateTokenizer",
    "apply_generation_chat_template",
    "controller_questions_and_answers",
    "require_chat_template_tokenizer",
    "require_train_template",
    "require_training_sampling",
    "resolve_algorithm_runtime_config",
]
