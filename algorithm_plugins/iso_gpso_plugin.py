"""Forward-KL self-distillation from verified online isomorphic examples."""

from __future__ import annotations

import random
from collections.abc import Mapping

import torch

from utils.algorithm_runtime import (
    AlgorithmRuntimeConfig,
    ChatTemplateTokenizer,
    apply_generation_chat_template,
    controller_questions_and_answers,
    resolve_teacher_sync_interval,
)
from utils.answer_verifier import verify_response_answer
from utils.prompts import render_prompt_from_record

from .algorithm_base import ObjectiveResult, Signals
from .kl_opsd_plugin import KLOPSDPlugin
from .pg_opd_plugin import _ACTOR_TRAJECTORY_SIGNALS
from .plugin_registry import register_plugin_class


@register_plugin_class
class IsoGPSOPlugin(KLOPSDPlugin):
    """Use verified original rollouts to condition the Teacher for analog rollouts."""

    computes_actor_question_accuracy = True

    def __init__(self, *, tokenizer: ChatTemplateTokenizer, runtime: AlgorithmRuntimeConfig) -> None:
        super().__init__(tokenizer=tokenizer, runtime=runtime)
        mode = runtime.algorithm.get("one_shot_format", "prompt")
        if not isinstance(mode, str) or mode not in {"prompt", "conversation"}:
            raise ValueError("algorithm.one_shot_format must be 'prompt' or 'conversation'.")
        self._one_shot_format = mode
        self._teacher_sync_interval = resolve_teacher_sync_interval(runtime.algorithm)
        self._seed = runtime.seed
        self._save_question_accuracy = runtime.save_question_accuracy
        self._target_questions = runtime.train_prompt_batch_size
        if (
            isinstance(self._target_questions, bool)
            or not isinstance(self._target_questions, int)
            or self._target_questions < 1
        ):
            raise ValueError("Iso-GPSO requires controller.train_prompt_batch_size to be a positive integer.")
        self._rollouts_per_question = self._sampling.get("n", 1)
        if (
            isinstance(self._rollouts_per_question, bool)
            or not isinstance(self._rollouts_per_question, int)
            or self._rollouts_per_question < 1
        ):
            raise ValueError("algorithm.training_sampling.n must be a positive integer.")

    def collect_actor_signals(self, signals: Signals) -> Signals:
        questions, _ = controller_questions_and_answers(signals)
        records = signals.get("records")
        if not isinstance(records, list) or len(records) != len(questions):
            raise ValueError("Iso-GPSO requires one aligned dataset record per original question.")
        prompts = []
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("Iso-GPSO dataset records must be mappings.")
            analog = record.get("analog")
            if not isinstance(analog, str) or not analog.strip():
                raise ValueError("Iso-GPSO requires a non-empty 'analog' string in every training record.")
            for question in (record["question"], analog):
                prompts.append(self._render_student_prompt({"question": question}))
        return {
            "prompts": prompts,
            "sampling": self._sampling,
            "return_signals": {**_ACTOR_TRAJECTORY_SIGNALS, "responses": True},
        }

    def finalize_actor_batch(self, signals: Signals) -> Signals:
        questions, answers = controller_questions_and_answers(signals)
        trajectory = signals["traj"]
        by_prompt: list[list[int]] = [[] for _ in range(2 * len(questions))]
        for index, prompt_index in enumerate(trajectory["prompt_indices"].tolist()):
            if not 0 <= prompt_index < len(by_prompt):
                raise ValueError("Iso-GPSO rollout index is outside the paired prompts.")
            by_prompt[prompt_index].append(index)
        if any(len(group) != self._rollouts_per_question for group in by_prompt):
            raise ValueError("Iso-GPSO requires n rollouts for both prompts of each question pair.")

        shard = signals.get("actor_shard", {})
        offset = shard.get("prompt_offset", 0)
        rng = random.Random(f"{self._seed}:{signals.get('batch_id', '')}:{offset}")
        selected: list[int] = []
        prompt_indices: list[int] = []
        teacher_prompts: list[str] = []
        accuracies: list[float] = []
        responses = trajectory["responses"]
        for question_index, answer in enumerate(answers):
            original_group, analog_group = by_prompt[2 * question_index:2 * question_index + 2]
            correct = [
                responses[index] for index in original_group
                if verify_response_answer(responses[index], str(answer))
            ]
            accuracies.append(len(correct) / self._rollouts_per_question)
            if not correct:
                continue
            record = signals["records"][question_index]
            rendered: dict[str, str] = {}
            for index in analog_group:
                example = rng.choice(correct)
                if example not in rendered:
                    rendered[example] = self._render_privileged_prompt({
                        "question": record["question"],
                        "analog": record["analog"],
                        "solution": self._example_text(example),
                    })
                selected.append(index)
                prompt_indices.append(question_index)
                teacher_prompts.append(rendered[example])
        signals["traj"] = {
            field: [trajectory[field][index] for index in selected]
            for field in ("prompt_ids", "response_ids")
        }
        signals["traj"]["prompt_indices"] = torch.tensor(prompt_indices, dtype=torch.long)
        signals["teacher_prompts"] = teacher_prompts
        if self._save_question_accuracy:
            signals["question_accuracies"] = accuracies
        return signals

    def _example_text(self, response: str) -> str:
        # Qwen drops the contents of a past assistant turn's reasoning block.
        # Keep the reasoning as normal text and remove generated delimiters.
        for token in (*getattr(self._tokenizer, "all_special_tokens", ()), "<think>", "</think>"):
            response = response.replace(token, "")
        return response.strip()

    def _render_teacher_record(self, record: Mapping[str, object]) -> str:
        example = render_prompt_from_record(self._train_template, {"question": record["question"]})
        target = render_prompt_from_record(self._train_template, {"question": record["analog"]})
        solution = record["solution"]
        if self._one_shot_format == "prompt":
            messages = [{"role": "user", "content": (
                f"Here is a solved example:\n\n{example}\n\nSolution:\n{solution}\n\n"
                f"Apply the same reasoning to solve the following problem:\n\n{target}"
            )}]
        else:
            messages = [
                {"role": "user", "content": example},
                {"role": "assistant", "content": solution},
                {"role": "user", "content": target},
            ]
        return apply_generation_chat_template(self._tokenizer, messages, enable_thinking=self._enable_thinking)

    def prepare_learner_batch(self, signals: Signals) -> Signals:
        # Only truncate after merging all Actor shards. Each accepted group
        # contributed exactly n rollouts, so the limit always falls between groups.
        limit = self._target_questions * self._rollouts_per_question
        trajectory = signals["traj"]
        signals["traj"] = {field: values[:limit] for field, values in trajectory.items()}
        signals["teacher_prompts"] = signals["teacher_prompts"][:limit]
        return signals

    def collect_learner_signals(self, signals: Signals) -> Signals:
        del signals
        return {}

    def compute_objective(self, batch: Signals) -> ObjectiveResult:
        result = super().compute_objective(batch)
        return ObjectiveResult(
            objective=result.objective,
            metrics={name.replace("kl_opsd_", "iso_gpso_", 1): value for name, value in result.metrics.items()},
            valid_token_count=result.valid_token_count,
        )

    def training_state_dict(self) -> dict[str, object]:
        return {
            **super().training_state_dict(),
            "algorithm": "iso_gpso",
            "one_shot_format": self._one_shot_format,
            "teacher_sync_interval": self._teacher_sync_interval,
            "rollouts_per_question": self._rollouts_per_question,
        }
