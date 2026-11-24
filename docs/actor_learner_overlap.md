# 按卡交接的 Actor / Learner 流水线

训练可在同一个全局 batch 内重叠生成和优化计算：本卡 Actor 完成任务、进入休眠并释放 GPU 执行权后，本卡 Learner 立即开始工作。其他卡继续生成。Learner 从共享队列逐条领取 rollout，不再固定平均分配样本。

```yaml
controller:
  max_inflight_batches: 1
  overlap_actor_learner: true
actor:
  evaluation:
    distribute_replies: true
```

两个开关独立。`overlap_actor_learner` 默认关闭，启用后使用按卡交接；`distribute_replies` 默认关闭，启用后训练中的周期评测及 `start_eval.sh` 独立评测均把同题的 8/16 条回复分散到多卡。所有要求的回复都会保留，没有额外采样或按完成速度筛选。

## 训练时序

1. Actor 沿用完整题组分片，生成本卡的所有回复。
2. 本卡 Actor 休眠、返回 GPU 执行权后，本卡 Learner 可立即运行。完整题组先完成 reward、优势或 TCPO 组内比例计算，再拆成独立 rollout 放入共享队列。
3. 空闲 Learner 每次领取一个 rollout，前向、反向并在本卡累积梯度。这个阶段使用 DeepSpeed `no_sync()`，包括信号收集在内均不调用跨卡集合通信。
4. 全部 Actor 完成、全局样本处理完毕后，各 Learner 同步一次实际优化器梯度缓冲区，按全局有效 token 数或序列数归一化，裁剪并更新一次参数。没有分到样本的 rank 也参与最终同步。
5. 发布同一版本的 LoRA，然后进入下一轮生成或周期评测。

整个 batch 使用同一行为策略。梯度按样本贡献求和，因此各卡处理数量不同也不改变样本权重。参数更新保持同步；此模式不会在其他 Actor 仍生成当前 batch 时更新策略。

当前支持 Actor/Learner 位于相同 GPU 集合、ZeRO stage 0、单个在途 batch。TCA、TCPO_v1、Forward-KL OPSD、Self-Shot/Ratio OPSD、TCDO 和 GRPO 有完整题组收集契约。需要全局 batch 决策的 RADO，以及成对提示后筛选的 Iso-GPSO，会在配置检查中明确拒绝。

## 评测分卡

每条回复有独立且可复现的 seed，按题目和回复编号轮转分配 GPU。每个 Actor 调用最多 64 条回复，仍使用原有并发、KV 预算和 CUDA Graph。收齐后按题目和回复编号恢复顺序，再计算 Avg@N、准确率、长度及截断率。AMC23 完成后才进入 AIME24。

周期评测缓存按完整题目保存，中断后可复用已完成的题目。独立评测的 rollout JSON 记录每条回复的 `actor_gpu_id`，可检查同题回复确实由多卡生成。

## 开启和验证

已提供 8 卡配置：

- `configs/training/tca/qwen3_4b_instruct_2507_overlap_8GPU.yaml`：32 题 × 8 回复，100 steps。
- `configs/training/tcpo_v1/qwen3_1p7b_overlap_8GPU.yaml`：16 题 × 16 回复，200 steps。

模型、LoRA、学习率和评测间隔继承原配置；训练上限 8192，评测上限 32768，Avg@16。其他模型配置也可添加上述两个开关。

8 卡服务器由用户操作，前台启动，不使用 tmux：

```bash
export FRUGALOPL_PYTHON="$(command -v python)"
export FRUGALOPL_SWANLAB_ENV_FILE=/private/path/swanlab.env
export FRUGALOPL_MODEL_PATH=/models/Qwen3-4B-Instruct-2507
bash scripts/start_train.sh --no-tmux configs/training/tca/qwen3_4b_instruct_2507_overlap_8GPU.yaml
```

双卡验证使用 tmux，测试文件仅保存在本地及测试服务器，不提交 GitHub：

```bash
bash scripts/start_train.sh --tmux tests/configs/overlap/tca_2gpu_r2.yaml
bash scripts/start_train.sh --tmux tests/configs/overlap/tcpo_2gpu.yaml
bash scripts/start_eval.sh --tmux tests/configs/overlap/eval_2gpu.yaml
```

双卡测试使用 Qwen3-1.7B No thinking、LoRA r=64/alpha=128，训练上限 1024、评测上限 128，用于验证计算、交接和回传。测试评测的准确率不作为正式 benchmark 成绩。双卡验证不代表已在 8 卡上实测。

SwanLab 的 `learner/actor_learner_overlap_seconds` 记录单个 batch 中 Learner 计算区间与其他 Actor 活动时间的重叠，`learner/per_rank_rollout_counts/*` 记录动态分工。运行目录 `overlap/step_*.json` 保存逐样本领取归属、策略版本和各卡时间区间，便于核对无重复、无遗漏以及真实交叠。

已完成的双卡验证包括：TCA 两次更新及每步 AMC23/AIME24 Avg@16；TCPO 每题 16 条回复、一次更新及 Avg@8；固定 rollout 的真实 BF16 DeepSpeed 梯度对照，覆盖 4/4、7/1、8/0 分工。梯度余弦均大于 0.9999，相对 L2 差异约 1.2%–1.3%；CPU 数值对照误差小于 5e-9。实际吞吐收益取决于生成与学习的耗时比例。

两次双卡短测分别耗时 23.88 秒和 21.36 秒，但生成量分别为 7712 和 6370 个 token，工作量不同，不能把 10.6% 的耗时差解释为加速收益。已通过逐样本时间区间确认实际交叠；加速幅度仍需固定工作量测试，也未在 8 卡上实测利用率。
