# TCPO_v1 与 TCA

四份正式配置均使用 Qwen3 No Thinking，单次同步全局 batch 为 **16 个问题 × 每题 16 个回复 = 256 条轨迹**。所有 Actor 同时生成，生成结束后同卡 Learner 同步优化，再进入下一批。

| 算法 | 模型 | GPU | Steps | 配置 |
| --- | --- | --- | --- | --- |
| TCPO_v1 | Qwen3-1.7B | 8 × 80 GB | 200 | `configs/training/tcpo_v1/qwen3_1p7b_200steps_8GPU.yaml` |
| TCPO_v1 | Qwen3-4B | 8 × 80 GB | 200 | `configs/training/tcpo_v1/qwen3_4b_200steps_8GPU.yaml` |
| TCA | Qwen3-1.7B | 4 × 80 GB | 100 | `configs/training/tca/qwen3_1p7b_100steps_4GPU.yaml` |
| TCA | Qwen3-4B | 4 × 80 GB | 100 | `configs/training/tca/qwen3_4b_100steps_4GPU.yaml` |

8 卡每卡 2 题 × 16 回复；4 卡每卡 4 题 × 16 回复。每个问题的 16 条回复始终属于同一组。

## 目标分布

Student 为 p，冻结的特权 Teacher 为 q。Teacher 使用同题标准解答构造 conversation self-shot 上下文，LoRA 禁用，基础参数冻结，不随 Student 更新。支持集和 TCA 目标为：

```text
S = {i: p_i >= 0.01 * max(p)}
r_i = normalize((p_i^2 / q_i) * 1[i in S])
```

实现使用 log 空间计算，在全词表上归一化；目标分布停止梯度。所有损失均为 Forward KL，没有额外 RL Loss。

**TCA（Teacher-Contrastive Amplification，教师对比放大）**：全部轨迹、全部有效回复 token 均优化 `KL(stopgrad(r) || p)`。答案正确性和截断状态不改变该目标。

**TCPO_v1**：先按截断优先，将轨迹互斥地划分为正确未截断 C、错误未截断 E、截断 T。截断不计入错误。

| 轨迹 | 目标 | α 默认值 |
| --- | --- | --- |
| C | `KL(stopgrad(q) || p)` | `tcpo_alpha_plus: 0.25` |
| E | `KL(stopgrad(r) || p)` | `tcpo_alpha_minus: 1.0` |
| T | `KL(stopgrad(r) || p)` | `tcpo_alpha_tru: 1.0` |

每题分别计算 `f_C=N_C/16`、`f_E=N_E/16`、`f_T=N_T/16`，三者之和为 1。每条轨迹额外乘所属类别的组内比例和 α：`w_j = alpha_class * f_class`。例如 8 正确、4 错误、4 截断，则对应每条轨迹权重为 0.125、0.25、0.25。

最终损失是所有轨迹加权 token 损失之和，除以**全部回复有效 token 总数**。不先对三个分支分别归约。组内比例在完整逻辑 batch 上计算，然后才按轨迹分配给各 Learner。

TCPO 使用 `tcpo_plausible_fraction: 0.01`；TCA 使用 `tca_plausible_fraction: 0.01`。

## 训练、评测与保存

- 训练集：`data/openthoughts_math30k.json`；seed=42；最大生成 8192；temperature=1.0、top-p=1.0。
- LoRA：r=64、alpha=128、dropout=0；学习率 5e-6；token mean；梯度裁剪 0.1。
- 初始及每 10 steps 评测一次，**先完整评测 AMC23，再评测 AIME24**。TCPO 覆盖 0、10…200；TCA 覆盖 0、10…100。
- AMC23 使用 `data/AMC-2023.json` 的全部 40 题，Avg@16 共 640 条回复；AIME24 使用 `data/AIME-2024.json` 的全部 30 题，Avg@16 共 480 条回复。
- 所有评测统一 temperature=0.6、top-p=0.95、最大生成 **32768**。全部 Actor 共同评测，每个 Actor 请求最多 64 条回复，并支持空闲 Actor 领取待运行批次。
- SwanLab 统一为 `qhe/TCDO`，按数据集分别记录 `avg_at_16`、截断率、截断数、回复数和平均长度；训练记录 Student/目标熵，TCPO 另记录三类比例、组内比例与实际轨迹权重。
- CUDA Graph 始终开启，`enforce_eager: false`；模型上下文窗口 40960。正式配置不启用旧的截断停训或自动减半权重流程，按固定的 200/100 步运行。
- 初始 LoRA 保存在 `checkpoints/step_0/adapter_lora`，每 10 步的评测 LoRA 保存在 `eval_adapters/step_<N>`；原始评测回复和分批恢复缓存保留。`save_ckpt_interval: 0` 避免另外累计完整 optimizer 检查点；临时 rolling 检查点用于精确恢复。

## 四台服务器的启动方式

每台服务器选择对应的一条命令。使用已有模型目录，先设置仓库外的 SwanLab 凭据文件路径：

```bash
export FRUGALOPL_SWANLAB_ENV_FILE=/your/private/path/swanlab.env
# 如需指定 Python：export FRUGALOPL_PYTHON=/your/env/bin/python
```

8 卡，TCPO_v1，1.7B：

```bash
export FRUGALOPL_MODEL_PATH=/your/existing/models/Qwen3-1.7B
bash scripts/start_train.sh --no-tmux configs/training/tcpo_v1/qwen3_1p7b_200steps_8GPU.yaml
```

8 卡，TCPO_v1，4B：

```bash
export FRUGALOPL_MODEL_PATH=/your/existing/models/Qwen3-4B
bash scripts/start_train.sh --no-tmux configs/training/tcpo_v1/qwen3_4b_200steps_8GPU.yaml
```

4 卡，TCA，1.7B：

```bash
export FRUGALOPL_MODEL_PATH=/your/existing/models/Qwen3-1.7B
bash scripts/start_train.sh --no-tmux configs/training/tca/qwen3_1p7b_100steps_4GPU.yaml
```

4 卡，TCA，4B：

```bash
export FRUGALOPL_MODEL_PATH=/your/existing/models/Qwen3-4B
bash scripts/start_train.sh --no-tmux configs/training/tca/qwen3_4b_100steps_4GPU.yaml
```

以上正式服务器命令均由 shell 前台直接运行 Python，不创建 tmux。双卡 4090 验证使用 `bash scripts/start_train.sh --tmux tests/configs/...yaml`；验证前台入口时使用外层 tmux 托管，内部入口不创建第二层 tmux。

## 实现版本与验证范围

新版 TCPO_v1 使用 `tcpo_objective_version: 2`，检查点绑定三类 α、支持集阈值和组内采样数。此前“正确零损失、截断靠近教师”的目标保留为 `TCDOPlugin`，历史 `qwen3_1p7b_100steps_8GPU.yaml` 改为明确引用该旧目标。新版拒绝将旧目标的 optimizer 状态当作同一个训练目标恢复。

CPU 检查覆盖组内三类互斥、16 回复分组、乱序对齐、加权 token 梯度、正确分支更新、截断分支 TCA、TCA 与原始 p²/q 更新一致，以及 4/8 Actor 按 AMC→AIME 顺序汇总真实题量和各自截断率。GPU 验证采用双卡、小 batch、短训练轨迹；正式 4/8 卡硬件由用户运行。

本次交付通过 81 项 CPU 检查，以及四种模型/算法组合各 2 步的真实双卡 4090 检查。每组在初始、第 1 步、第 2 步均由两个 Actor 顺序完成两个短诊断数据集的 Avg@16 评测，并保存 LoRA；TCPO 的正确、错误、截断三个分支均实际执行。1.7B 的 TCPO 还实际使用每题 16 回复的完整分组。1.7B 测试保留 32768 生成上限与 40960 上下文；4B 为适配 24 GB 显存使用 1024 生成上限与 8192 上下文，正式 4B 配置仍为 32768 / 40960。诊断回复较短，这些检查验证启动、目标与训练评测链路，不是正式 100/200 步性能评测。
