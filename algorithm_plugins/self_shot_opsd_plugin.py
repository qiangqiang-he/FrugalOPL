"""Forward-KL OPSD with the same solved question supplied as a one-shot."""

from __future__ import annotations

from collections.abc import Mapping

from utils.algorithm_runtime import AlgorithmRuntimeConfig, ChatTemplateTokenizer, apply_generation_chat_template
from utils.prompts import render_prompt_from_record

from .algorithm_base import ObjectiveResult, Signals
from .kl_opsd_plugin import KLOPSDPlugin
from .plugin_registry import register_plugin_class


@register_plugin_class
class SelfShotOPSDPlugin(KLOPSDPlugin):
    """Change only the frozen Teacher's context; reuse the full KL objective.

    ``prompt`` embeds a solved example and the repeated question in one user
    message. ``conversation`` assigns the example solution to an actual past
    assistant turn. Both question turns use the unchanged Student template.
    The inherited prompt limiter may shorten only the example solution.
    """

    def __init__(self, *, tokenizer: ChatTemplateTokenizer, runtime: AlgorithmRuntimeConfig) -> None:
        super().__init__(tokenizer=tokenizer, runtime=runtime)
        mode = runtime.algorithm.get("self_shot_format")
        if not isinstance(mode, str) or mode not in {"prompt", "conversation"}:
            raise ValueError("algorithm.self_shot_format must be 'prompt' or 'conversation'.")
        self._self_shot_format = mode

    def _render_teacher_record(self, record: Mapping[str, object]) -> str:
        question = render_prompt_from_record(self._train_template, record)
        solution = record["solution"]
        if self._self_shot_format == "prompt":
            messages = [{"role": "user", "content": (
                "Here is a solved example:\n\n"
                f"{question}\n\nSolution:\n{solution}\n\n"
                f"Now solve the same problem:\n\n{question}"
            )}]
        else:
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": solution},
                {"role": "user", "content": question},
            ]
        return apply_generation_chat_template(
            self._tokenizer, messages, enable_thinking=self._enable_thinking,
        )

    def compute_objective(self, batch: Signals) -> ObjectiveResult:
        result = super().compute_objective(batch)
        return ObjectiveResult(
            objective=result.objective,
            metrics={key.replace("kl_opsd_", "self_shot_opsd_", 1): value for key, value in result.metrics.items()},
            valid_token_count=result.valid_token_count,
        )

    def training_state_dict(self) -> dict[str, object]:
        return {**super().training_state_dict(), "algorithm": "self_shot_opsd",
                "self_shot_format": self._self_shot_format}


__all__ = ["SelfShotOPSDPlugin"]
