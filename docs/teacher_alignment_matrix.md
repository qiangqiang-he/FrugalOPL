# FKL、Clip-OPSD、TRA：八模型训练配置

三种算法正式命名为 **FKL**、**Clip-OPSD** 和 **Teacher reverse align（TRA）**。新配置、插件、训练状态和 SwanLab 指标使用这些名称；旧 TCA 入口继续兼容已有运行和检查点。

所有新实验记录到公开项目 [qhe/TCPO](https://swanlab.cn/@qhe/TCPO)。`run.group` 使用模型名；`run.name` 包含算法、模型、thinking 模式、训练步数和 GPU 数。DeepSeek 的 run name 分别使用 `deepseek-1p5b`、`deepseek-7b`，省略模式标签，group 保留完整模型名。

[全部 24 条启动命令、SwanLab 隐私边界及 TRA 指数参数](teacher_alignment_launch_commands.md)。

三组共用同一冻结基础 Teacher：关闭 Teacher 的 LoRA，并将该题解答作为已完成的 assistant 示例，再重复 Student 的问题。Student 提示和特权 Teacher 上下文在三种算法间一致。

| 算法 | GPU 数 | 目标 |
| --- | ---: | --- |
| FKL | 4 | 最小化普通 Forward KL：`KL(q || p)`，无 CLIP |
| Clip-OPSD | 4 | 对每个词表项的 `q_i log(q_i / p_i)` 做上界截断，epsilon=`1e-7` |
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

Qwen3、Qwen3.5、Gemma4 使用 No thinking；DeepSeek Distill 使用其原生推理模板。Qwen3.5 的 LoRA 包含线性注意力投影；Gemma 的 LoRA 仅作用于实际存在的文本注意力和 MLP 层。

## 24 份配置

| 模型 / SwanLab group | FKL · 4 GPU | Clip-OPSD · 4 GPU | TRA · 8 GPU |
| --- | --- | --- | --- |
| Qwen3-1.7B | [配置](../configs/training/fkl/qwen3_1p7b_100steps_4GPU.yaml) | [配置](../configs/training/clip_opsd/qwen3_1p7b_100steps_4GPU.yaml) | [配置](../configs/training/tra/qwen3_1p7b_100steps_8GPU.yaml) |
| Qwen3-4B | [配置](../configs/training/fkl/qwen3_4b_100steps_4GPU.yaml) | [配置](../configs/training/clip_opsd/qwen3_4b_100steps_4GPU.yaml) | [配置](../configs/training/tra/qwen3_4b_100steps_8GPU.yaml) |
| Qwen3.5-4B | [配置](../configs/training/fkl/qwen3_5_4b_100steps_4GPU.yaml) | [配置](../configs/training/clip_opsd/qwen3_5_4b_100steps_4GPU.yaml) | [配置](../configs/training/tra/qwen3_5_4b_100steps_8GPU.yaml) |
| Qwen3.5-2B | [配置](../configs/training/fkl/qwen3_5_2b_100steps_4GPU.yaml) | [配置](../configs/training/clip_opsd/qwen3_5_2b_100steps_4GPU.yaml) | [配置](../configs/training/tra/qwen3_5_2b_100steps_8GPU.yaml) |
| Gemma4-E2B-it | [配置](../configs/training/fkl/gemma4_e2b_it_100steps_4GPU.yaml) | [配置](../configs/training/clip_opsd/gemma4_e2b_it_100steps_4GPU.yaml) | [配置](../configs/training/tra/gemma4_e2b_it_100steps_8GPU.yaml) |
| Gemma4-E4B-it | [配置](../configs/training/fkl/gemma4_e4b_it_100steps_4GPU.yaml) | [配置](../configs/training/clip_opsd/gemma4_e4b_it_100steps_4GPU.yaml) | [配置](../configs/training/tra/gemma4_e4b_it_100steps_8GPU.yaml) |
| DeepSeek-R1-Distill-Qwen-1.5B | [配置](../configs/training/fkl/deepseek_1p5b_100steps_4GPU.yaml) | [配置](../configs/training/clip_opsd/deepseek_1p5b_100steps_4GPU.yaml) | [配置](../configs/training/tra/deepseek_1p5b_100steps_8GPU.yaml) |
| DeepSeek-R1-Distill-Qwen-7B | [配置](../configs/training/fkl/deepseek_7b_100steps_4GPU.yaml) | [配置](../configs/training/clip_opsd/deepseek_7b_100steps_4GPU.yaml) | [配置](../configs/training/tra/deepseek_7b_100steps_8GPU.yaml) |

例如：`FKL_deepseek-1p5b_100steps_4gpus`、`Clip-OPSD_Qwen3.5-4B_no-thinking_100steps_4gpus`、`TRA_Gemma4-E4B-it_no-thinking_100steps_8gpus`。

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

## 基础矩阵验证（2026-11-25，提交 b1033d7）

CPU 检查 **114 项通过**，覆盖全部配置、算法注册、Controller 启动、三种目标的独立梯度核对、原生模型 LoRA 结构及聊天模板。远程双 RTX 4090 上 **24/24 组通过**：每组使用完整 BF16 基础权重，执行 2 次实际优化更新，单批 2 题 × 4 回复。

每组都完成两轮 AMC23 → AIME24 双卡评测，核实更新后的 LoRA 热加载、各数据集截断率上传、评测期间下一批 rollout 预取，以及全部评测结束后的 Learner 放行。梯度和优化器状态均为有限值；每个目标层的 LoRA A/B 形状均与基础权重匹配，所有 B 权重和模块更新均非零。服务端按 group 筛选核实了全部八个模型分组及公开项目 TCPO。

正式 4/8 卡配置通过解析与协议检查；实际 GPU 验证使用上述双卡轻量覆盖，未在 4/8 卡服务器上运行完整 100 步或 8192/32768 长度实验。95 个生产源码及配置文件的指纹与远程实测版本一致。

### 双卡吞吐

下表每格为 **Actor 生成 / 训练流水线**，单位均为 response tok/s，两批合并后按总 token 数除以总耗时计算。流水线耗时从训练任务派发到更新结果回收，包含显存交接和等待。生成上限 256，每批最多 2048 个回复 token。

| 模型 | LoRA 目标层数 | FKL | Clip-OPSD | TRA |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-1.7B | 196 | 1289.4 / 292.6 | 1271.5 / 290.4 | 1277.4 / 291.0 |
| Qwen3-4B | 252 | 685.9 / 127.4 | 688.6 / 124.6 | 685.8 / 125.9 |
| Qwen3.5-4B | 248 | 598.7 / 84.3 | 609.6 / 85.4 | 609.7 / 85.0 |
| Qwen3.5-2B | 186 | 934.4 / 136.0 | 1100.0 / 139.3 | 1109.5 / 132.6 |
| Gemma4-E2B-it | 205 | 560.0 / 104.0 | 934.0 / 114.3 | 931.3 / 115.5 |
| Gemma4-E4B-it | 258 | 563.7 / 75.9 | 564.1 / 74.8 | 564.0 / 75.1 |
| DeepSeek-R1-Distill-Qwen-1.5B | 196 | 1314.2 / 300.7 | 1304.5 / 299.3 | 1307.5 / 298.4 |
| DeepSeek-R1-Distill-Qwen-7B | 196 | 454.1 / 77.8 | 454.9 / 79.0 | 454.6 / 78.9 |

短批次结果存在首批波动。例如 Gemma4-E2B-it 的 FKL 首批生成 427.5、第二批 811.6 tok/s；同模型 Clip-OPSD、TRA 第二批分别为 822.6、823.7 tok/s。逐批生成、Learner 和流水线吞吐保留在本地 CSV/JSON 中。

Qwen3-1.7B、DeepSeek-1.5B 的测试基座常驻显存，其余模型使用冻结基座 CPU/GPU 交接。正式 80GB 配置采用基座常驻，吞吐需按正式 batch 和长度实测。Qwen3.5 Learner 在本次环境中使用原生 PyTorch GatedDeltaNet 路径，环境未安装 `fla`、`causal_conv1d`。

GPU 验证环境：PyTorch `2.10.0+cu129`、Transformers `5.9.0`、PEFT `0.21.0`、vLLM `0.19.1`、DeepSpeed `0.19.7`、SwanLab `0.10.4`。

完整证据保留在本地 `tests/outputs/teacher_alignment_matrix_20261125/`：`gpu_validation_summary.json`、`throughput.csv`、各组 LoRA/梯度检查、云端分组核验及日志。测试结束后已清理临时 LoRA、检查点和优化器状态，八个基础模型保留，双卡显存已释放。

该轮 24 条旧云端测试记录已按用户要求删除，本地指标和报告保留。隐私修复后的验证另存于 `tests/outputs/swanlab_privacy_20261125/`。
