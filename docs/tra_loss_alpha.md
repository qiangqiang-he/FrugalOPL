# TRA 损失系数：alpha=0.5，n=0.1

TRA 增加 `algorithm.tra_loss_alpha`，取值范围 `[0,1]`，默认 `1.0`。目标分布与当前 TRA 相同，alpha 只在 KL 损失外相乘：

\[
r_i\propto p_i(p_i/q_i)^n\mathbf 1[p_i\ge0.01p_{\max}],
\qquad L=\alpha\,\mathrm{KL}(\operatorname{stopgrad}(r)\Vert p).
\]

n 控制目标分布，alpha 控制损失和对应梯度的整体强度。默认 alpha=1 与原损失、梯度保持一致；alpha=0.5 将损失及梯度缩放为原来的一半。alpha=0 时，TRA 不贡献梯度，并跳过优化器更新，避免历史动量继续改变参数。

```yaml
algorithm:
  tra_loss_alpha: 0.5
  tra_ratio_power: 0.1
```

SwanLab 公开配置记录 alpha；每步记录原始 `tra_raw_loss`、缩放后的 `tra_loss`、`tra_loss_alpha` 和 `tra_ratio_power`。所有损失指标按原有有效回复 token 归约。检查点保存 alpha，恢复时核对一致；此前未保存 alpha 的检查点按 alpha=1 兼容恢复。

三份新配置均为八卡、No thinking、同题 solution one-shot、100 次优化更新、全局 batch 32 题 × 每题 8 回复。沿用 Student top-512、支持集阈值 0.01、冻结基座 Teacher、学习率 5e-6，以及 LoRA r=64／alpha=128。训练生成上限 8192，温度 1.0、top-p 1.0；评测上限 32768，温度 0.6、top-p 0.95，初始及每 10 步依次在 AMC23、AIME24 上做 Avg@16 评测并保存 LoRA。

## 编号及八卡启动命令：无 tmux

| 编号 | 模型 | GPU 数 | 损失 alpha | n | 状态 |
| --- | --- | ---: | ---: | ---: | --- |
| **35** | Qwen3-1.7B | 8 | 0.5 | 0.1 | 待运行 |
| **36** | Qwen3-4B | 8 | 0.5 | 0.1 | 待运行 |
| **37** | Qwen3-8B | 8 | 0.5 | 0.1 | 待运行 |

在训练服务器激活环境并进入仓库，设置模型目录与仓库外的凭据文件：

```bash
export FRUGALOPL_PYTHON="$(command -v python)"
export FRUGALOPL_MODELS_ROOT="/path/to/models"
export FRUGALOPL_SWANLAB_ENV_FILE="/path/outside/repo/swanlab.env"
```

### 编号 35｜Qwen3-1.7B｜TRA alpha=0.5，n=0.1｜8 卡

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-1.7B"
# 编号 35
bash scripts/start_train.sh --no-tmux configs/training/tra/qwen3_1p7b_n0p1_alpha0p5_100steps_8GPU.yaml
```

### 编号 36｜Qwen3-4B｜TRA alpha=0.5，n=0.1｜8 卡

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-4B"
# 编号 36
bash scripts/start_train.sh --no-tmux configs/training/tra/qwen3_4b_n0p1_alpha0p5_100steps_8GPU.yaml
```

### 编号 37｜Qwen3-8B｜TRA alpha=0.5，n=0.1｜8 卡

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-8B"
# 编号 37
bash scripts/start_train.sh --no-tmux configs/training/tra/qwen3_8b_n0p1_alpha0p5_100steps_8GPU.yaml
```

## 双卡验证命令：有 tmux

新损失缩放链路用 Qwen3-1.7B 在双卡上验证；4B、8B 的模型结构和 LoRA 目标层沿用既有验证。轻量测试使用 2 题 × 4 回复、2 次实际更新，训练生成上限 256、短评测上限 128。测试配置、脚本和输出只保留在本地 `tests/`，不上传 GitHub。

```bash
bash scripts/start_train.sh --tmux tests/configs/tra_alpha/qwen3_1p7b.yaml
```

正式八卡配置由用户启动；双卡验证用于核对算法链路、alpha 缩放、检查点、LoRA 更新与评测调度。

2026-11-27：**198 项 WSL CPU 检查通过**，覆盖三个新配置、独立梯度、默认 alpha=1 的逐值等价、alpha=0 的动量零更新、旧检查点兼容、TRA 特权变体及分布式调度回归。

Qwen3-1.7B 已用双卡完成 2 次真实优化更新及两轮短评测。第一步 `tra_raw_loss=0.0011718099359753799`、`tra_loss=0.0005859049679876899`，第二步也保持精确 0.5 倍关系；记录的 n=0.1、alpha=0.5 与检查点一致。196 个 LoRA 目标层均有有限非零更新；CUDA Graph、LoRA 热加载、逐数据集截断率和评测预取屏障通过。测试使用离线 SwanLab，未新增云端测试 run。

测试证据保存在 Windows 与 WSL 的 `tests/outputs/tra_alpha_20261127/`；临时 checkpoint、LoRA 及评测 adapter 已清理，双卡已释放。4B、8B 的新配置通过 CPU 优化链路验证，其模型与 LoRA 接入沿用此前测试；本次新增损失缩放的 GPU 实测使用 1.7B。
