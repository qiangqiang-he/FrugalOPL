# 8 卡 Instruct TCA

两份配置均面向 **8 × 80 GB**，Actor 与 Learner 位于同一组 8 张卡，单次仅有一个同步 batch：全部 Actor 生成结束，再由全部 Learner 优化。

| 模型 | 配置 |
| --- | --- |
| `Qwen/Qwen3-4B-Instruct-2507` | `configs/training/tca/qwen3_4b_instruct_2507_100steps_8GPU.yaml` |
| `allenai/Olmo-3-7B-Instruct` | `configs/training/tca/olmo3_7b_instruct_100steps_8GPU.yaml` |

## 算法和训练设置

- TCA：全部轨迹拟合停止梯度的 `normalize(p²/q)`；Student、冻结的特权 Teacher 均在 Student 每个位置选出的 **Top-512** 上归一化，再保留 `p_i >= 0.01 × p_max` 的支持集。使用普通 Forward KL，没有 CLIP 或额外答案奖励损失。
- Teacher 为同一基础模型，禁用 LoRA；以同题标准解答构造 conversation self-shot。只更新 Student LoRA。
- 每批 **32 个问题 × 每题 8 个回复 = 256 条轨迹**，8 卡每卡分配 4 个问题 / 32 条回复；训练 **100 次更新**。
- OpenThoughts Math30K；LoRA `r=64, alpha=128`；学习率 `5e-6`；token mean；梯度裁剪 `0.1`。
- 训练最大生成 **8192**，temperature `1.0`、top-p `1.0`；整段回复计算，`kl_top_k: 512`。
- 初始和每 **10** 步，先完整评测 **AMC23**，再评测 **AIME24**；两者均为 **Avg@16**、temperature `0.6`、top-p `0.95`、最大生成 **32768**。
- 所有 8 个 Actor 共同评测，单个 Actor 请求上限为 64 条回复；SwanLab 为 `qhe/TCDO`，分别记录每个数据集的准确率、截断率和平均长度，并记录训练截断率、熵和梯度。
- `enforce_eager: false`，使用 CUDA Graph。OLMo 使用 8 条并发序列及 0.70 Actor 显存预算，适配其 MHA 与滑动/全局混合注意力，给同卡 Learner 留出实际运行空间。
- 初始 LoRA 及每次评测的 LoRA 均保存；不额外累计周期性完整 optimizer 检查点，保留滚动恢复状态。

[Qwen 模型卡](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)说明该版本仅支持 non-thinking；这里使用的 OLMo 是 [Instruct 版本](https://huggingface.co/allenai/Olmo-3-7B-Instruct)，不是 Think checkpoint。使用模型各自的官方 chat template。

## 8 卡启动：前台 Python，不使用 tmux

在两台服务器上分别运行对应配置。未设置 `FRUGALOPL_MODEL_PATH` 时使用配置中的官方 Hugging Face ID；已有本地权重时，将该变量设置为对应模型的目录。

```bash
export FRUGALOPL_SWANLAB_ENV_FILE=/your/private/path/swanlab.env
# export FRUGALOPL_PYTHON=/your/env/bin/python
```

Qwen3-4B-Instruct-2507：

```bash
# 可选：export FRUGALOPL_MODEL_PATH=/your/models/Qwen3-4B-Instruct-2507
bash scripts/start_train.sh --no-tmux configs/training/tca/qwen3_4b_instruct_2507_100steps_8GPU.yaml
```

Olmo 3 7B Instruct：

```bash
# 可选：export FRUGALOPL_MODEL_PATH=/your/models/Olmo-3-7B-Instruct
bash scripts/start_train.sh --no-tmux configs/training/tca/olmo3_7b_instruct_100steps_8GPU.yaml
```

使用本地模型目录时，两组实验各自设置正确的 `FRUGALOPL_MODEL_PATH`，它会覆盖 YAML 中的模型 ID。数据文件沿用 `data/openthoughts_math30k.json`、`data/AMC-2023.json`、`data/AIME-2024.json`。

## 双卡验证：使用 tmux

本地验证配置保留在 `tests/`，不提交到 GitHub：

```bash
bash scripts/start_train.sh --tmux tests/configs/tca_instruct/qwen_2gpu.yaml
bash scripts/start_train.sh --tmux tests/configs/tca_instruct/olmo_2gpu.yaml
```

小规模验证为 2 题 × 8 回复、2 次更新，检查 LoRA 更新、Teacher 上下文、训练后加载新 LoRA、AMC→AIME 评测与分数据集截断率。诊断评测使用短长度限制，只验证运行流程，不作为模型性能评测。

Qwen 的双卡验证使用两卡共享 Actor/Learner；7B 的两份 BF16 权重及优化器不能同时常驻一张 24 GB 卡，双卡验证因此使用 GPU 0 生成、GPU 1 优化。正式 OLMo 配置仍是 8 张 80 GB 卡上同卡共享、同步交替。

OLMo 3 的 Transformers 支持要求至少 4.57.0。验证环境为 PyTorch `2.10.0+cu129`、Transformers `5.9.0`、vLLM `0.19.1`、PEFT `0.21.0`。

本次通过 **65 项 CPU 检查**，包括 OLMo 3 混合注意力下的真实 LoRA / TCA 反向传播。两种完整模型均通过上述两步 GPU 训练及每步 AMC→AIME 评测，未出现训练或评测异常：

| 模型 | 实际更新次数 | 发生变化的 LoRA 张量 | SwanLab |
| --- | --- | --- | --- |
| Qwen3-4B-Instruct-2507 | 2 | 504 / 504 | [双卡检查](https://swanlab.cn/@qhe/TCDO/runs/t2lhy9n6/chart) |
| Olmo 3 7B Instruct | 2 | 448 / 448 | [两卡分角色检查](https://swanlab.cn/@qhe/TCDO/runs/7ogau5lk/chart) |

模型权重按官方 Hugging Face SHA256 校验。远程模型目录为 `/models/Qwen3-4B-Instruct-2507` 和 `/models/Olmo-3-7B-Instruct`。测试产物备份回 Windows 工作区后清理服务器上的临时 checkpoint；模型权重保留。
