# TCPO_v1

TCPO_v1 是当前双卡实验中三分支目标的正式名称。`TCDOPlugin` 保留为早期名称，`TCPOV1Plugin` 使用相同目标，并能读取早期检查点；参数名继续沿用 `tcdo_*`。

对于一条 Student 轨迹，截断优先于答案正确性：

| 轨迹 | 训练目标 |
| --- | --- |
| 未截断且答案正确 | 零损失 |
| 未截断且答案错误 | 对停止梯度的 `r ∝ p²/q · 1[p ≥ 0.01 p_max]` 做 `KL(r || p)` |
| 截断 | 普通 `KL(q || p)` |

`q` 来自冻结的基础模型，使用同题标准解答作为 conversation self-shot 特权上下文。目标在全词表构造。所有回复的有效 token 一起归约，包括正确轨迹的零损失 token；两个非零分支不分别求均值。没有额外 RL Loss。

```yaml
tcdo_plausible_fraction: 0.01
tcdo_incorrect_weight: 1.0
tcdo_truncated_weight: 1.0
```

## 八卡配置

入口：`configs/training/tcpo_v1/qwen3_1p7b_100steps_8GPU.yaml`。服务器按 8 张 80 GB GPU 配置。

| 项目 | 设置 |
| --- | --- |
| 模型 | Qwen3-1.7B，`enable_thinking: false` |
| 每步全局 batch | 32 个问题 × 8 个回复 = 256；每卡 4 题 × 8 个回复 |
| 执行 | 8 个 TP=1 Actor 同时生成，然后同卡的 8 个 Learner 同步优化；一次一个 batch |
| LoRA | r=64，alpha=128，dropout=0 |
| 训练 | 每轮最多 100 步，最大生成 8192，temperature=1，top-p=1 |
| 评测 | AIME24，Avg@16，30×16=480 条；temperature=0.6，top-p=0.95，最大生成 32768 |
| 评测频率 | 初始及每 10 步；所有 Actor 共同评测，每次 Actor 请求最多 64 条 |
| 学习率 | 5e-6；token mean；梯度裁剪 0.1 |
| Teacher | 冻结基础模型，conversation self-shot，教师不随 Student 更新 |
| 运行 | CUDA Graph；Actor/Learner 常驻模型，Actor 在 Learner 阶段释放 KV 缓存 |
| SwanLab | `qhe/TCDO`，记录准确率、每数据集截断率/截断数、生成长度和训练熵 |

## 启动

八卡服务器由用户操作，前台启动，不使用 tmux。模型路径指向服务器上已有的 Qwen3-1.7B：

```bash
export FRUGALOPL_MODEL_PATH=/your/existing/model/path/Qwen3-1.7B
export FRUGALOPL_SWANLAB_ENV_FILE=/your/private/path/swanlab.env
bash scripts/start_train.sh --no-tmux configs/training/tcpo_v1/qwen3_1p7b_100steps_8GPU.yaml
```

如需指定 Python，可另设 `FRUGALOPL_PYTHON`。本地双卡验证使用 `--tmux` 和 `tests/configs/` 下的小规模配置。

当前双卡长期实验按用户指定，中间评测使用 Avg@8，第 100 步使用 Avg@16，生成上限仍为 20480。可在 `actor.evaluation` 下设置：

```yaml
rollouts_per_prompt: 8
final_rollouts_per_prompt: 16
```

`final_rollouts_per_prompt` 仅在达到 `controller.max_steps` 的最终评测时生效，Actor 按该次请求采样，结果记录实际的 `avg_at_8` 或 `avg_at_16`。不设置该选项则各次评测均使用 `rollouts_per_prompt`；八卡正式配置仍全程 Avg@16。双卡此前已经完成的 Avg@16 结果保留，第 50 步起中间评测改用 Avg@8。

## 自适应权重与保存

`run.tcpo_v1_tuning.enabled: true` 启用与当前实验相同的调权规则。每轮从同一 seed=42 的初始模型重新开始；权重改变后不接着优化上一轮的 LoRA。

- 第 10 步起，AIME24 截断率严格超过 50%：停止该轮，错误轨迹权重减半，重新训练。
- 完成 100 步且第 10–100 步各次评测截断率均严格低于 10%：截断轨迹权重减半，重新训练。
- 其余完成 100 步的情况：本次调权结束。
- 完整评测前，如果已确认截断数除以全部预定 480 条已经超过 50%，可提前停止；未完成点不报告为完整 Avg@16。

默认最多 6 轮。达到轮数预算仍需调整时，清楚记录 `review_window_complete`；增加 `max_trials` 后重新执行同一启动命令即可继续。初始评测只跑第一轮，其余轮共享这次基线。设 `run.tcpo_v1_tuning.enabled: false` 可只启动配置中的单轮。

`outputs/<run.name>_campaign/manifest.json` 保存每轮配置、权重、评测及决策。各轮 `checkpoints/step_0/adapter_lora` 保留初始 LoRA，`eval_adapters/step_10` 至 `step_100` 保存评测 LoRA；原始回复与分批评测缓存保留。每轮退出且 Worker 已关闭后，删除该轮不再使用的 optimizer 临时检查点。

双卡 4090 已完成两轮真实验证，每轮 1 步、4 题 × 2 个回复，覆盖正确、错误和截断三个分支。测试完成双卡并行评测、CUDA Graph、40960 上下文窗口与 32768 生成上限，并验证了截断权重从 1 减半到 0.5 后以相同初始 LoRA 重启、仅首轮评测基线、两轮使用不同 SwanLab 实验及清理 optimizer 检查点。验证采用 2 道短 AIME24 题，回复约 600 tokens；32K 是配置上限。另有 57 项 CPU 检查通过，包括新旧目标逐参数一致、八卡配置协议检查及中间/最终评测采样数切换。额外的双卡 2 步测试在初始、第 1 步、第 2 步分别实际生成 2、2、4 条评测回复，验证最终评测自动增加采样数，三个评测点均由两个 Actor 共同完成。

八卡 80 GB 配置由用户运行，尚无八卡硬件实测结果。
