# standard_FKL：标准 OPSD 形式的单轮 solution 特权

`standard_FKL` 与 [standard_OPSD](standard_opsd.md) 使用完全相同的 Student 和 Teacher 提示。Teacher 只有一个 user turn，按“当前题目一次＋reference solution＋transition”拼接；特权中额外添加 solution，不重复题目构造 one-shot。

唯一的损失差别是关闭 CLIP：`algorithm.kl_token_clip: null`。在 Student top-512 上，将 Student 和冻结 Teacher 的分布分别归一化，优化普通 Forward KL：

\[
L=\operatorname{mean}_{\text{有效回复 token}}
  \sum_{i\in\operatorname{Top512}(p)}q_i\log\frac{q_i}{p_i}.
\]

`StandardFKLPlugin` 复用已有的单轮 Teacher 模板、solution 缩短策略和 KL 训练链路。Teacher 使用关闭 LoRA 的冻结基座，不参与梯度更新。配置中的 CLIP 必须为 null；省略该配置时，插件也强制关闭 CLIP。检查点记录 `algorithm: standard_FKL`、`teacher_context: single_turn_reference_solution`、`token_clip: null`，避免与带 CLIP 或 one-shot 的检查点混用。

三组均为八卡、No thinking、100 次实际优化更新、batch 32 题 × 每题 8 回复。训练生成上限 8192，温度 1.0、top-p 1.0。初始及每 10 步依次评测 AMC23、AIME24，Avg@16、温度 0.6、top-p 0.95、生成上限 32768，并保存 LoRA。学习率 5e-6，LoRA r=64、alpha=128。

继续使用 Actor/Learner 共卡调度、同题评测回复跨卡分配、评测期间下一 batch rollout 预取和 Learner 释放屏障。SwanLab 记录到公开 `TCPO` Project，group 为模型名称；过滤本地路径与个人信息。记录 `standard_fkl_loss`、`standard_fkl_unclipped_kl`、恒为零的 `standard_fkl_clip_fraction`、熵、生成长度及各评测数据集的截断率。损失、辅助 KL 和熵统一按有效回复 token 归约；关闭 CLIP 后，两条 KL 损失指标数值一致。

## 编号与八卡启动命令：无 tmux

编号与[实验优先级与启动清单](实验优先级与启动清单.md)一致：

| 编号 | 模型 | 算法 | GPU 数 | 状态 |
| --- | --- | --- | ---: | --- |
| **27** | Qwen3-1.7B | standard_FKL | 8 | 待运行 |
| **28** | Qwen3-4B | standard_FKL | 8 | 待运行 |
| **29** | Qwen3-8B | standard_FKL | 8 | 待运行 |

激活训练环境并进入项目根目录，设置仓库外的凭据文件及模型根目录：

```bash
export FRUGALOPL_PYTHON="$(command -v python)"
export FRUGALOPL_MODELS_ROOT="/path/to/models"
export FRUGALOPL_SWANLAB_ENV_FILE="/path/outside/repo/swanlab.env"
```

### 编号 27｜Qwen3-1.7B｜standard_FKL｜8 卡

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-1.7B"
# 编号 27
bash scripts/start_train.sh --no-tmux configs/training/standard_fkl/qwen3_1p7b_100steps_8GPU.yaml
```

### 编号 28｜Qwen3-4B｜standard_FKL｜8 卡

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-4B"
# 编号 28
bash scripts/start_train.sh --no-tmux configs/training/standard_fkl/qwen3_4b_100steps_8GPU.yaml
```

### 编号 29｜Qwen3-8B｜standard_FKL｜8 卡

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-8B"
# 编号 29
bash scripts/start_train.sh --no-tmux configs/training/standard_fkl/qwen3_8b_100steps_8GPU.yaml
```

## 双卡验证：有 tmux

测试配置仅保留在本地 `tests/`，不上传 GitHub。每组使用 2 题 × 4 回复、2 次实际优化更新，训练生成上限 256、评测生成上限 128，使用短算术 fixture 验证训练及逐数据集评测链路。正式八卡配置的评测上限为 32768。

```bash
bash scripts/start_train.sh --tmux tests/configs/standard_fkl/qwen3_1p7b.yaml
bash scripts/start_train.sh --tmux tests/configs/standard_fkl/qwen3_4b.yaml
bash scripts/start_train.sh --tmux tests/configs/standard_fkl/qwen3_8b.yaml
```

CPU 检查覆盖与 standard_OPSD 的提示逐字一致性、独立普通 FKL 梯度、配置继承、检查点隔离、Student adapter 更新、冻结 Teacher、隐私和启动命令。双卡验证检查实际 Teacher prompt、无 CLIP、LoRA 更新、CUDA Graph、跨卡评测、预取屏障及截断率记录；正式八卡配置由用户启动。

2026-11-26：**132 项 WSL CPU 检查通过**，三组均完成真实双卡测试：

| 编号 | 模型 | 实际 GPU 数 | 实际更新数 | 完整覆盖并更新的 LoRA 目标层 |
| --- | --- | ---: | ---: | ---: |
| 27 | Qwen3-1.7B | 2 | 2 | 196 |
| 28 | Qwen3-4B | 2 | 2 | 252 |
| 29 | Qwen3-8B | 2 | 2 | 252 |

每组审计 16 条实际 Teacher prompt，均为一个 user turn、题目一次、完整 solution 及 transition。三组 CLIP 比例均为零，普通 KL loss 与未截断 KL 指标数值一致；梯度和优化器状态有限，所有目标 LoRA 层均有非零更新。两轮双数据集评测链路、逐数据集截断率、CUDA Graph、LoRA 热加载、跨卡回复分配和评测预取屏障均通过。365 个交付源码／配置文件与远程测试版本一致。

测试使用离线 SwanLab，没有新增云端测试 run；临时 checkpoint、LoRA 及评测 adapter 已清理，双卡已释放。测试证据同步保存在 Windows 和 WSL 的 `tests/outputs/standard_fkl_20261126/`，测试文件不上传 GitHub。以上为双卡短生成验证，正式八卡及 32K 评测配置由用户启动。
