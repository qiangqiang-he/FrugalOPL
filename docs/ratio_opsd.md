# DeepSeek-R1 Ratio-OPSD：8 卡实验

本实验拟合停止梯度的完整词表目标：

\[
r_i=\frac{(p_i^2/q_i)\,\mathbf{1}[p_i\ge0.1\max_j p_j]}
{\sum_j(p_j^2/q_j)\,\mathbf{1}[p_j\ge0.1\max_k p_k]},
\qquad L=\mathrm{KL}(\mathrm{stopgrad}(r)\Vert p).
\]

保留之前 Ratio 实验的 0.1 支持集规则。目标按 log 空间计算，避免直接除以很小的 Teacher 概率。Student 和 Teacher 的分布温度均为 1.0。没有 CLIP 截断、额外前缀权重或答案奖励损失。

Teacher 为关闭 LoRA 的冻结基础模型；特权上下文为同一题目的 `user → assistant(数据集解答) → user` 对话。Student 只看题目。基础参数冻结，仅训练 LoRA，r=64、alpha=128。

## 配置和启动

两个入口共享 `configs/common/ratio_opsd_8GPU.yaml`，默认 8×80 GB。

```bash
conda activate post_train
export FRUGALOPL_PYTHON="$(command -v python)"
export FRUGALOPL_SWANLAB_ENV_FILE="$HOME/.config/frugalopl/swanlab.env"
export MODEL_ROOT="$HOME/projects/models"  # 改成服务器实际模型父目录。
```

1.5B 的启动命令：

```bash
export FRUGALOPL_MODEL_PATH="$MODEL_ROOT/DeepSeek-R1-Distill-Qwen-1.5B"
bash scripts/start_train.sh --no-tmux configs/training/ratio_opsd/deepseek_r1_distill_qwen_1p5b_100steps_8GPU.yaml
```

上一场结束后，7B 的启动命令：

```bash
export FRUGALOPL_MODEL_PATH="$MODEL_ROOT/DeepSeek-R1-Distill-Qwen-7B"
bash scripts/start_train.sh --no-tmux configs/training/ratio_opsd/deepseek_r1_distill_qwen_7b_100steps_8GPU.yaml
```

`FRUGALOPL_MODEL_PATH` 会覆盖 YAML 中的 `run.policy.model_name_or_path`，检查本地目录存在，并把实际路径保存到本次运行的合并配置。两种启动模式都使用该路径；未设置时使用 YAML 默认路径。数据文件需要预先存在：`data/openthoughts_math30k.json`、`data/AMC-2023.json`、`data/AIME-2024.json`。Redis 默认 `127.0.0.1:6379`。

8 卡使用 `--no-tmux`：shell 脚本直接在前台 `exec` Python，保留进程 PID、信号和退出码，不创建 tmux 会话。输出同时显示在终端并写入启动日志；前台退出码由调用方或调度器接收，不生成 tmux 的 `.status` 文件。

双卡 4090 必须由 tmux 托管，使用默认模式或显式 `--tmux`：

```bash
bash scripts/start_train.sh --tmux tests/configs/ratio_opsd/deepseek_1p5b_2GPU_smoke.yaml
```

启动器会输出 tmux 会话名、日志和 `.status` 文件路径。独立评估脚本 `scripts/start_eval.sh` 使用相同的 `--tmux` / `--no-tmux` 参数约定。若在双卡验证前台入口，先创建外层 tmux 托管任务，再在该会话中调用 `--no-tmux`，避免双卡任务依赖 SSH 会话存活。

| 设置 | 值 |
|---|---|
| 优化步数 | 100 |
| 全局训练 batch | 32 题 × 每题 8 个回复，共 256 条 |
| 调度 | 同步，最多 1 个 batch 在途 |
| 训练最大回复长度 | 20480 tokens（20K） |
| 训练采样 | temperature=1.0，top-p=1.0，无 top-k 截断 |
| 评估最大回复长度 | 32768 tokens（32K） |
| 评估采样 | temperature=0.6，top-p=0.95，每题 16 个回复 |
| 评估时点 | 0、10、20、…、100 |
| checkpoint 时点 | 0、10、20、…、100，共 11 份 |
| 学习率 | 5e-6，constant |
| SwanLab 项目 | qhe/TCDO |

Actor 和 Learner 共用全部 8 张 GPU。一个训练 batch 按问题分片给全部 Actor；每卡 4 题、32 条回复。全部 rollout 结束后，Actor 进入 vLLM level-1 sleep，八个 Learner rank 共同完成一次优化，再发布新 LoRA。CUDA Graph 始终开启。

## 并行评测和截断率

每个评估时点先完整评测 AMC23，再完整评测 AIME24。每个数据集都将题目分配到全部 8 个 Actor；每个 Actor 同时最多处理一个评测请求，单次最多 64 条回复。较快的 Actor 继续处理自己剩余的题目，最后按原始题目顺序汇总，逐项核对 GPU 和 adapter 版本。

完整 AMC23 为 40×16=640 条回复，AIME24 为 30×16=480 条。`Avg@16` 是每题 16 次正确率再按题目平均；独立保留 Pass@16。

SwanLab 分别记录：

- `evaluation/amc23/avg_at_16`、`evaluation/aime24/avg_at_16`。
- `evaluation/amc23/truncation_rate`、`evaluation/aime24/truncation_rate`。
- 每个数据集的 `response_count`、`truncated_response_count`、`actor_count` 和 `avg_response_length`。
- `actor/truncation_rate`：训练 rollout 的截断比例；以及 `ratio_opsd/student_entropy_nats` 等分布指标。

截断标志来自 vLLM 的完成原因；使用 token 长度作为旧接口的回退依据。截断率分母为该数据集全部回复，包含正确且被截断的回复。

每次评测保存逐样本结果；checkpoint 在 `outputs/<run.name>/checkpoints/step_<N>/` 下，包含 `adapter_lora/` 和训练状态。`learner.checkpoint.max_keep` 只影响临时发布目录，不删除上述 11 份定期 checkpoint。初始 checkpoint 单独保存，不改变初始 Actor 使用的基础策略。

## 验证范围

双卡 4090 使用 1.5B、小 batch 和缩短回复的测试配置，验证真实 LoRA 优化、CUDA Graph、共卡 sleep/wake、并行评测和 checkpoint 保存。7B 仅检查配置，不在双卡加载；8 卡机器由用户运行正式实验。
