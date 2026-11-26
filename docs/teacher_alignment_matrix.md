# FKL、Clip-OPSD、TRA：当前三模型训练配置

三种算法正式命名为 **FKL**、**Clip-OPSD** 和 **Teacher reverse align（TRA）**。新配置、插件、训练状态和 SwanLab 指标使用这些名称；旧 TCA 入口继续兼容已有运行和检查点。

所有新实验记录到公开项目 [qhe/TCPO](https://swanlab.cn/@qhe/TCPO)。`run.group` 使用模型名；`run.name` 包含算法、模型、thinking 模式、训练步数和 GPU 数。

[基础实验及 TRA 特权对照启动命令、SwanLab 隐私边界及 TRA 指数参数](teacher_alignment_launch_commands.md)。当前优先级与连续实验编号见 [实验优先级与启动清单](实验优先级与启动清单.md)。

三组共用同一冻结基础 Teacher：关闭 Teacher 的 LoRA，并将该题解答作为已完成的 assistant 示例，再重复 Student 的问题。Student 提示和特权 Teacher 上下文在三种算法间一致。

| 算法 | GPU 数 | 目标 |
| --- | ---: | --- |
| FKL | Qwen3-4B / 8B：8；其余：4 | 最小化普通 Forward KL：`KL(q || p)`，无 CLIP |
| Clip-OPSD | Qwen3-4B / 8B：8；其余：4 | 对每个词表项的 `q_i log(q_i / p_i)` 做上界截断，epsilon=`1e-7` |
| TRA | 8 | 拟合停止梯度的 `r ∝ p × (p/q)^n × 1[p ≥ 0.01 p_max]`，默认 `n=1` |

三者默认在 Student 每个位置选择的 top-512 上，将 Student/Teacher 分布分别归一化后构造损失，按原有 token 方式归约。训练记录的 Student/Teacher 熵也在该支持集上计算。TRA 的 `tra_plausible_fraction` 默认 `0.01`，`tra_ratio_power` 对应指数 `n`、默认 `1.0`。

| 设置 | 默认值 |
| --- | --- |
| 总更新数 | 100 |
| 全局 batch | 32 个问题 × 每题 8 个回复，共 256 条 |
| 训练最大生成长度 | 8192 |
| 训练采样 | temperature=1.0，top-p=1.0 |
| 评测 | 初始及每 10 步，AMC23 完成后再 AIME24，Avg@16 |
| 评测最大生成长度 | 32768 |
| 评测采样 | temperature=0.6，top-p=0.95 |
| 评测请求批次 | 64 个回复，同题回复分散至各 Actor |
| LoRA | r=64，alpha=128 |
| 学习率 | 5e-6 |
| 检查点 | 初始及每 10 步保存，另外保留滚动恢复状态和评测 LoRA 快照 |
| 运行方式 | 每卡 Actor/Learner 交接；评测末尾预取下一训练 batch，全部评测完成后才开放 Learner |
| vLLM | CUDA Graph 开启，enforce_eager=false |

当前三个 Qwen3 模型均使用 No thinking，LoRA 作用于文本注意力及 MLP 的七类投影层。

## 九组主实验配置

| 模型 / SwanLab group | FKL · 4/8 GPU | Clip-OPSD · 4/8 GPU | TRA · 8 GPU |
| --- | --- | --- | --- |
| Qwen3-1.7B | [配置](../configs/training/fkl/qwen3_1p7b_100steps_4GPU.yaml) | [配置](../configs/training/clip_opsd/qwen3_1p7b_100steps_4GPU.yaml) | [配置](../configs/training/tra/qwen3_1p7b_100steps_8GPU.yaml) |
| Qwen3-4B | [8 卡配置](../configs/training/fkl/qwen3_4b_100steps_8GPU.yaml) | [8 卡配置](../configs/training/clip_opsd/qwen3_4b_100steps_8GPU.yaml) | [配置](../configs/training/tra/qwen3_4b_100steps_8GPU.yaml) |
| Qwen3-8B | [8 卡配置](../configs/training/fkl/qwen3_8b_100steps_8GPU.yaml) | [8 卡配置](../configs/training/clip_opsd/qwen3_8b_100steps_8GPU.yaml) | [配置](../configs/training/tra/qwen3_8b_100steps_8GPU.yaml) |

例如：`FKL_Qwen3-4B_no-thinking_100steps_8gpus`、`Clip-OPSD_Qwen3-8B_no-thinking_100steps_8gpus`。

Qwen3-1.7B 的基础 TRA 现明确命名为 `tra_self_shot`，现有配置路径保留。另有两份八卡特权对照：[tra_random_shot](../configs/training/tra_random_shot/qwen3_1p7b_100steps_8GPU.yaml)、[tra_true_answer](../configs/training/tra_true_answer/qwen3_1p7b_100steps_8GPU.yaml)。三者优化相同目标，只改变冻结 Teacher 的特权上下文。

Qwen3-4B、Qwen3-8B 的同题 solution TRA 和 Clip-OPSD 已于 2026-11-26 由用户确认完成。新增两份八卡 TRA 不匹配 solution one-shot 配置：[Qwen3-4B](../configs/training/tra_mismatch_solution/qwen3_4b_100steps_8GPU.yaml)、[Qwen3-8B](../configs/training/tra_mismatch_solution/qwen3_8b_100steps_8GPU.yaml)。示例从训练集中严格排除当前题后随机抽取，保留示例题与其 solution 的配对，其他训练及评测超参数与原 TRA 一致。

三份同题 solution TRA 指数对照均为 n=0.5；Qwen3-1.7B 使用四卡，Qwen3-4B、Qwen3-8B 使用八卡，其余参数沿用对应模型的 n=1 配置：

| 模型 | n=0.5 配置 | 清单编号／状态 |
| --- | --- | --- |
| Qwen3-1.7B | [4 卡配置](../configs/training/tra/qwen3_1p7b_n0p5_100steps_4GPU.yaml) | 17／已启动 |
| Qwen3-4B | [8 卡配置](../configs/training/tra/qwen3_4b_n0p5_100steps_8GPU.yaml) | 18 |
| Qwen3-8B | [8 卡配置](../configs/training/tra/qwen3_8b_n0p5_100steps_8GPU.yaml) | 19 |

新增 Qwen3-1.7B 的两组四卡空答案对照，均为 No thinking、n=1、100 steps、batch 16×8、训练 8192／评测 32768、Avg@16：

| 实验 | 配置 | 清单编号／状态 |
| --- | --- | --- |
| TRA-match-answer：提取不到自身答案时用 `""` | [4 卡配置](../configs/training/tra_match_answer/qwen3_1p7b_empty_fallback_100steps_4GPU.yaml) | 20／待运行 |
| TRA-mismatch-answer：无可用不匹配答案时用 `""` | [4 卡配置](../configs/training/tra_mismatch_answer/qwen3_1p7b_empty_fallback_100steps_4GPU.yaml) | 21／待运行 |

## 正式服务器启动：无 tmux

先激活训练环境。模型路径通过环境变量覆盖 YAML 的 Hugging Face 名称；SwanLab 凭据文件位于仓库外。

```bash
export FRUGALOPL_PYTHON="$(command -v python)"
export FRUGALOPL_MODEL_PATH="/path/to/Qwen3-1.7B"
export FRUGALOPL_SWANLAB_ENV_FILE="/path/outside/repo/swanlab.env"

# 4 卡 FKL
bash scripts/start_train.sh --no-tmux configs/training/fkl/qwen3_1p7b_100steps_4GPU.yaml

# 4 卡 Clip-OPSD（单独启动的任务）
bash scripts/start_train.sh --no-tmux configs/training/clip_opsd/qwen3_1p7b_100steps_4GPU.yaml

# 8 卡 TRA（单独启动的任务）
bash scripts/start_train.sh --no-tmux configs/training/tra/qwen3_1p7b_100steps_8GPU.yaml
```

三个命令按服务器资源分别启动。脚本前台直接执行 Python，由服务器调度系统管理。

## 双卡验证：使用 tmux

本工作区保留 24 份 `tests/configs/teacher_alignment/` 轻量配置，每份继承对应正式配置，仅缩小到双卡、2 题 × 4 回复、2 次更新、训练生成上限 256。每次更新后，以每数据集 1 题、Avg@2、生成上限 128 检查 AMC23 → AIME24 评测、LoRA 热加载、数据预取和指标上传。

```bash
bash scripts/start_train.sh --tmux tests/configs/teacher_alignment/fkl_qwen3_1p7b.yaml
bash scripts/start_train.sh --tmux tests/configs/teacher_alignment/clip_opsd_qwen3_1p7b.yaml
bash scripts/start_train.sh --tmux tests/configs/teacher_alignment/tra_qwen3_1p7b.yaml
```

双卡大模型验证使用冻结基座 CPU/GPU 交接，正式 80GB 配置保留基座常驻。测试配置、代码、日志和结果仅保留在本地 `tests/`，不提交 GitHub。每组验证完毕清理临时适配器和检查点，基础模型保留。

## 验证记录

当前三个模型的基础算法已有双卡轻量验证记录，每组执行两次实际更新并检查评测、LoRA 热加载及各数据集截断率。证据保存在本地 `tests/outputs/teacher_alignment_matrix_20261125/` 和 `tests/outputs/stage1_priority_20261125/`。

新增两组不匹配 solution TRA 已通过 113 项 CPU 检查，覆盖严格排除同题、示例题与 solution 配对、跨 Worker 一致性、恢复检查及实际 CPU Learner 更新。本轮远程双卡不可用，未进行新的 GPU 实测；证据保存在本地 `tests/outputs/tra_mismatched_solution_20261126/`。

正式八卡训练由用户在服务器启动。
