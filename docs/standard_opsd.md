# standard_OPSD：单轮 reference solution 特权与 CLIP

`standard_OPSD` 使用官方 OPSD `reason_first=False` 的提示构造方式，复用仓库已有的 `KL_OPSD_TEACHER_PROMPT`。参考：[官方 data_collator.py](https://github.com/siyan-zhao/OPSD/blob/main/data_collator.py)。本实验使用冻结 base Teacher、Student top-512 和逐词表项 CLIP。Qwen3-1.7B 的 No thinking 配置按官方 README 使用 epsilon=1e-6。

Student 只读取当前题目及解题指令。Teacher 使用**一个 user turn**，内容依次为：

1. 当前题目，只拼接一次。
2. `Reference Solution Begin/End` 标记包围的当前题目 solution。
3. 官方 transition 指令：理解参考解法后独立推理，给出相同最终答案，并使用 boxed 格式。

随后应用同一模型的 chat template 和 No thinking 设置，并对 Student 已生成的回复进行 Teacher 前向评分。特权部分只增加 solution；没有额外的示例 question，也不构造 user／assistant／user 三轮 one-shot。过长的 Teacher prompt 只缩短 solution，保留完整题目、transition 和 assistant generation suffix。

算法为逐词表项上界截断的 Forward KL。在 Student top-512 上分别归一化 p、q，使用：

\[
L=\operatorname{mean}_{\text{有效回复 token}}
  \sum_{i\in\operatorname{Top512}(p)}\min\left(q_i\log\frac{q_i}{p_i},\epsilon\right).
\]

当前配置的 epsilon：Qwen3-1.7B、4B 为 `1e-6`，Qwen3-8B 为 `1e-7`。官方 [README 的 No thinking 小节](https://github.com/siyan-zhao/OPSD#non-thinking-mode)将 1.7B 标注为 `--jsd_token_clip 1e-6`；通用 [run_opsd_1b.sh](https://github.com/siyan-zhao/OPSD/blob/main/scripts/run_opsd_1b.sh)则使用 `0.05`，且未关闭 thinking。本次三组配置均按 No thinking 对齐：1.7B、4B 使用 `1e-6`，8B 使用 `1e-7`。4B、8B 分别见官方 [4B No thinking 脚本](https://github.com/siyan-zhao/OPSD/blob/main/scripts/run_opsd_4b_nonthink.sh)、[8B No thinking 脚本](https://github.com/siyan-zhao/OPSD/blob/main/scripts/run_opsd_8b_nonthink.sh)。

Teacher 使用冻结基座、关闭 LoRA 和梯度；Student 使用 LoRA 接受更新。没有奖励项。`StandardOPSDPlugin` 直接复用已有 KL 评分与优化链路，仅选择单轮上下文并强制启用 CLIP。检查点记录 `algorithm: standard_OPSD` 和 `teacher_context: single_turn_reference_solution`，防止与旧 one-shot 目标混用。

Qwen3-1.7B 使用四卡，Qwen3-4B、Qwen3-8B 使用八卡；三组均为 No thinking、100 次优化更新、batch 32 题 × 每题 8 回复。训练生成上限 8192；初始及每 10 步依次评测 AMC23、AIME24，Avg@16、温度 0.6、top-p 0.95、生成上限 32768，并保存 LoRA。学习率 5e-6，LoRA r=64、alpha=128。沿用 Actor/Learner 共卡调度、评测回复跨卡分配、评测期间 rollout 预取及 Learner 释放屏障。

SwanLab 使用公开的 `TCPO` Project，group 为模型名；继续过滤本地路径、提示正文和其他个人信息。记录 `standard_opsd_loss`、`standard_opsd_unclipped_kl`、`standard_opsd_clip_fraction`、熵、训练生成长度，以及各评测数据集的截断率。

## 四卡／八卡启动命令：无 tmux

编号与[实验优先级与启动清单](实验优先级与启动清单.md)一致：

| 编号 | 模型 | 算法 | GPU 数 | CLIP epsilon | 状态 |
| --- | --- | --- | ---: | --- | --- |
| **24** | Qwen3-1.7B | standard_OPSD | 4 | `1e-6` | 待运行 |
| **25** | Qwen3-4B | standard_OPSD | 8 | `1e-6` | 待运行 |
| **26** | Qwen3-8B | standard_OPSD | 8 | `1e-7` | 待运行 |

先激活训练环境并进入项目根目录，设置仓库外的凭据文件和模型根目录。

```bash
export FRUGALOPL_PYTHON="$(command -v python)"
export FRUGALOPL_MODELS_ROOT="/path/to/models"
export FRUGALOPL_SWANLAB_ENV_FILE="/path/outside/repo/swanlab.env"
```

每份配置在对应卡数的服务器上独立启动：

### 编号 24｜Qwen3-1.7B｜standard_OPSD｜4 卡

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-1.7B"
# 编号 24
bash scripts/start_train.sh --no-tmux configs/training/standard_opsd/qwen3_1p7b_100steps_4GPU.yaml
```

### 编号 25｜Qwen3-4B｜standard_OPSD｜8 卡

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-4B"
# 编号 25
bash scripts/start_train.sh --no-tmux configs/training/standard_opsd/qwen3_4b_100steps_8GPU.yaml
```

### 编号 26｜Qwen3-8B｜standard_OPSD｜8 卡

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-8B"
# 编号 26
bash scripts/start_train.sh --no-tmux configs/training/standard_opsd/qwen3_8b_100steps_8GPU.yaml
```

## 双卡验证：有 tmux

测试配置仅保留在本地 `tests/`，不上传 GitHub。每组使用 2 题 × 4 回复、2 次实际优化更新；训练生成上限 256、评测上限 128，使用短算术 fixture 验证 AMC23／AIME24 命名的评测链路。正式 32K 配置由 CPU 继承检查核对，短测试的正确率不用于评价算法性能。

```bash
bash scripts/start_train.sh --tmux tests/configs/standard_opsd_clip1e6/qwen3_1p7b.yaml
bash scripts/start_train.sh --tmux tests/configs/standard_opsd_clip1e6/qwen3_4b.yaml
bash scripts/start_train.sh --tmux tests/configs/standard_opsd/qwen3_8b.yaml
```

2026-11-26 早先版本（正式配置均八卡、CLIP=1e-7）：WSL CPU 检查 **117 项通过**。以下三组均已按顺序完成真实双卡验证：

| 模型 | 实际 GPU 数 | 实际更新数 | 完整覆盖并更新的 LoRA 目标层 |
| --- | ---: | ---: | ---: |
| Qwen3-1.7B | 2 | 2 | 196 |
| Qwen3-4B | 2 | 2 | 252 |
| Qwen3-8B | 2 | 2 | 252 |

每组审计 16 条 rollout 的真实 Teacher prompt：一个 user turn、题目一次、完整 solution 及 transition。CLIP 实际生效，梯度与优化器状态均有限；所有 LoRA 目标层覆盖、尺寸和非零更新通过检查。三组 CUDA Graph、LoRA 热加载、两轮双数据集评测链路、逐数据集截断率、跨卡回复分配及评测预取屏障均通过。当时的 360 个交付源码／配置文件与该次双卡验证版本一致；此记录对应上述早先版本。

测试使用离线 SwanLab，未新增云端测试 run。临时 checkpoint、LoRA 与评测 adapter 已清理，双卡已释放；测试脚本和完整证据只保留在本地 `tests/`，不上传 GitHub。正式四卡／八卡配置由用户操作，以上实测均为双卡验证。

### 本次官方 No thinking CLIP 对齐

2026-11-26：编号 24 改为四卡、CLIP=`1e-6`；编号 25 保持八卡、CLIP 改为 `1e-6`；编号 26 保持八卡，并显式设置官方 CLIP=`1e-7`。编号不变，其他训练与评测参数保持原设置。

更新后的 **132 项 WSL CPU 检查通过**。Qwen3-1.7B、4B 均以 `1e-6` 完成双卡小批量复测，每组 2 题 × 4 回复、2 次真实优化更新、两轮短评测；196／252 个 LoRA 目标层均有有限非零更新。CLIP 生效、CUDA Graph、LoRA 热加载、逐数据集截断率和评测预取屏障均通过。365 个交付源码／配置文件与此次远程测试版本一致。

Qwen3-8B 的有效 CLIP 参数仍为 `1e-7`，沿用上述已通过的双卡记录，本次没有重跑该组。新测试证据保存在本地 `tests/outputs/standard_opsd_clip1e6_20261126/`；临时检查点已清理，双卡已释放。
