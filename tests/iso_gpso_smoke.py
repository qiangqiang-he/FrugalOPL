"""Exercise Iso-GPSO filtering and KL scoring without stepping an optimizer."""

import argparse
import copy
import json
import tempfile
import time
from pathlib import Path

import torch

from algorithm_plugins import plugin_factory_from_config
from algorithm_plugins.algorithm_base import _learner_plugin_scope
from engine.actor import Actor, _merge_actor_shards, load_merged_config
from engine.controller import Controller
from engine.learner import Learner, _lora_adapter_root, _move_tensors_to_device


CONFIG = "configs/training/iso_gpso/qwen3_1p7b_rtx5000_2_shared_batched_4GPU.yaml"


def signals_for(records, *, offset=0, shard=0, shards=1, total=None):
    return {
        "batch_id": "train-000001",
        "questions": [record["question"] for record in records],
        "answers": [record["answer"] for record in records],
        "records": records,
        "actor_shard": {
            "index": shard, "count": shards, "prompt_offset": offset,
            "global_prompt_count": len(records) if total is None else total,
        },
    }


class TextTokenizer:
    all_special_tokens = ["<|im_end|>"]

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(f'{message["role"]}: {message["content"]}' for message in messages)

    def __call__(self, prompt, **kwargs):
        return {"input_ids": prompt.split()}


def check_filtering(config):
    config["controller"]["train_prompt_batch_size"] = 2
    config["algorithm"]["one_shot_format"] = "conversation"
    config["algorithm"]["training_sampling"]["n"] = 2
    config["save_question_accuracy"] = True
    factory = plugin_factory_from_config(config)
    assert Controller(config, plugin_factory=factory).train_candidate_count == 3
    plugin = factory(TextTokenizer())
    rows = [
        {"question": f"original-{index}", "answer": "7", "analog": f"analog-{index}"}
        for index in range(3)
    ]

    def actor_shard(offset, shard, solutions):
        source = signals_for(rows[offset:offset + len(solutions)], offset=offset, shard=shard, shards=2, total=3)
        request = plugin.collect_actor_signals(source)
        assert len(request["prompts"]) == 2 * len(solutions)
        texts = [value for pair in solutions for value in (*pair, r"Wrong \boxed{900}", "analog output")]
        actor = Actor(config, plugin=plugin)
        prepared = actor._prepare_training_batch(source)
        return actor._finish_training_batch(prepared, {
            "prompt_ids": [torch.tensor([index + 1]) for index in range(len(texts))],
            "response_ids": [torch.tensor([index + 10]) for index in range(len(texts))],
            "responses": texts,
            "prompt_indices": torch.arange(2 * len(solutions)).repeat_interleave(2),
        }, {})

    valid = (r"<think>reason</think> Final \boxed{7}<|im_end|>", r"Wrong \boxed{9}")
    invalid = (r"Wrong \boxed{9}", r"Wrong \boxed{900}")
    mixed = _merge_actor_shards([actor_shard(0, 0, [invalid, valid]), actor_shard(2, 1, [valid])])
    prepared = plugin.prepare_learner_batch(mixed)
    assert prepared["traj"]["prompt_indices"].tolist() == [1, 1, 2, 2]
    assert prepared["question_accuracies"] == [0., .5, .5]
    assert all("assistant: reason Final" in prompt and "900" not in prompt for prompt in prepared["teacher_prompts"])
    overflow = _merge_actor_shards([actor_shard(0, 0, [valid, valid]), actor_shard(2, 1, [valid])])
    assert plugin.prepare_learner_batch(overflow)["traj"]["prompt_indices"].tolist() == [0, 0, 1, 1]
    empty = _merge_actor_shards([actor_shard(0, 0, [invalid, invalid]), actor_shard(2, 1, [invalid])])
    result = Learner(config, plugin=plugin, device="cpu").process_training_batch(empty)
    assert result["optimizer_steps"] == result["global_step"] == result["rollout_count"] == 0
    with tempfile.TemporaryDirectory() as output:
        config["run"]["storage"]["output_root"] = output
        controller = Controller(config, plugin_factory=factory)
        controller._confirmed_batch_sequence = 2
        controller._write_progress(started=time.time(), completed_steps=0)
        resumed = Controller(config, plugin_factory=factory)
        assert resumed._restore_run_progress() == 0
        assert resumed._confirmed_batch_sequence == 2 and resumed._next_train_offset == 6
    print("CPU filtering, global cap, reasoning, and empty batch passed", flush=True)


def check_models(config, model_path, gpu):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cpu" if gpu is None else "cuda:0")
    dtype = torch.float32 if gpu is None else torch.bfloat16
    config["run"]["policy"]["model_name_or_path"] = model_path
    config["run"]["policy"]["dtype"] = "float32" if gpu is None else "bfloat16"
    config["run"]["policy"]["lora"].update(rank=8, alpha=16)
    config["actor"]["lora"]["max_rank"] = 8
    config["controller"]["train_prompt_batch_size"] = 1
    config["algorithm"]["training_sampling"].update(n=2, max_tokens=384, temperature=0.3)
    config["algorithm"]["training_sampling"].pop("top_k")
    config["algorithm"]["teacher_sync_interval"] = 2
    config["algorithm"]["kl_token_clip"] = None
    config["actor"]["engine"].update(
        enforce_eager=False, gpu_memory_utilization=0.25, max_model_len=2048,
        max_num_seqs=8, compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8]},
    )
    if gpu is not None:
        config["actor"]["gpu_ids"] = config["learner"]["gpu_ids"] = [gpu]
    records = [{
        "question": "What is 2 plus 3?", "answer": "5", "analog": "What is 4 plus 7?",
    }, {
        "question": "What is 2 plus 5?", "answer": "7", "analog": "What is 4 plus 3?",
    }]
    source = signals_for(records)
    with tempfile.TemporaryDirectory() as root:
        config["run"]["storage"].update(output_root=root, lora_adapter_root=root)
        factory = plugin_factory_from_config(config)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        plugin = factory(tokenizer)
        actor = Actor(config, plugin=plugin, gpu_id=gpu)
        if gpu is not None:
            try:
                actor.initialize_engine()
                actor.start_generation_loop()
                generated = actor.process_training_batch(source)
            finally:
                actor.close()
            assert generated["actor_metrics"]["batch_rollout_count"] == 8
        else:
            prepared = actor._prepare_training_batch(source)
            encoded = tokenizer(prepared.prompts, add_special_tokens=False)["input_ids"]
            texts = [r"Add the integers. \boxed{5}", r"\boxed{5}", r"\boxed{11}", r"\boxed{11}",
                     r"Add the integers. \boxed{7}", r"\boxed{7}", r"\boxed{7}", r"\boxed{7}"]
            generated = actor._finish_training_batch(prepared, {
                "prompt_ids": [torch.tensor(ids) for ids in encoded for _ in range(2)],
                "response_ids": [torch.tensor(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]],
                "responses": texts,
                "prompt_indices": torch.arange(4).repeat_interleave(2),
            }, {})
        logical = plugin.prepare_learner_batch(generated)
        assert logical["traj"]["prompt_indices"].numel() == 2, generated["actor_metrics"] if gpu is not None else generated
        print(f"Paired rollouts and global cap passed; retained {len(logical['teacher_prompts'])} trajectories", flush=True)

        learner = Learner(config, plugin=plugin, device=device, gpu_id=gpu)
        learner.tokenizer = tokenizer
        base = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, attn_implementation="sdpa", local_files_only=True,
        )
        learner.model = learner._attach_or_restore_lora(base)
        learner._initialize_forward_kl_teacher_adapter()
        learner.model.to(device)
        batch = _move_tensors_to_device(learner.build_micro_batches(logical)[0], device)

        def score():
            with _learner_plugin_scope(learner, phase="objective"):
                result = plugin.compute_objective(batch)
            assert result.valid_token_count > 0 and torch.isfinite(result.objective)
            return float(result.objective.detach())

        frozen = score()
        # Publish a deliberately different snapshot. No backwards or optimizer
        # step is needed to check that the Teacher reads the published version.
        default = [p for n, p in learner.model.named_parameters() if "lora_B.default" in n]
        with torch.no_grad():
            for parameter in default:
                parameter.normal_(std=0.01)
        learner.global_step = 2
        directory = _lora_adapter_root(config) / "step_2" / "adapter_lora"
        learner.model.save_pretrained(directory, selected_adapters=["default"])
        synced = score()
        assert learner._teacher_adapter_step == 2 and learner.model.active_adapter == "default"
        teacher = [p for n, p in learner.model.named_parameters() if "lora_B.forward_kl_teacher" in n]
        assert teacher and all(not p.requires_grad for p in teacher)
        assert all(torch.equal(a, b) for a, b in zip(default, teacher, strict=True))
        learner.global_step = 3
        assert learner._forward_kl_teacher_step() == 2
        learner.teacher_sync_interval = -1
        assert score() != synced
        learner.teacher_sync_interval = 0
        assert score() == synced
        print(json.dumps({"frozen_base_objective": frozen, "synced_objective": synced, "optimizer_steps": 0}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    config = load_merged_config(CONFIG)
    check_filtering(copy.deepcopy(config))
    if args.model:
        torch.set_num_threads(4)
        check_models(config, args.model, args.gpu)
