# 八模型 × 三算法及 TRA 特权对照：全部启动命令

分支：`TCPO_v1`。FKL、Clip-OPSD 使用 **4 卡**，Teacher reverse align（TRA）使用 **8 卡**；正式服务器均通过 `--no-tmux` 在前台启动。以下包含 24 个基础实验和 2 个新增 TRA 特权对照，分别在相应卡数的服务器上选择运行。

所有配置默认训练 100 次更新，全局 batch 为 32 题 × 每题 8 个回复，训练生成上限 8192；初始及每 10 步依次评测 AMC23、AIME24，Avg@16，生成上限 32768。LoRA r=64、alpha=128，分布损失使用 Student top-512。

## SwanLab 记录范围与名称

云端只上传选定的实验参数和数值指标：模型公开名称、算法、thinking 模式、采样与优化参数、LoRA 参数、准确率、长度、熵、各数据集截断率及吞吐。配置中的本地模型路径、数据/输出路径、Redis 地址、凭据、提示词内容均不会复制到云端。

SDK 的终端采集、运行环境、主机信息、Git、依赖快照和 SwanLab 本地目录采集关闭；数值型 GPU/CPU 利用率监控保留。异常详情保留在本地日志，云端只记录固定说明。此边界通过 SDK 的 `Settings` 配置，在初始化前生效，参考 [官方设置说明](https://docs.swanlab.cn/api/py-settings.html)。本地 YAML 和启动日志仍保留完整内容，便于调试。

项目为公开的 [qhe/TCPO](https://swanlab.cn/@qhe/TCPO)，group 使用模型名称。名称示例：

- `FKL_Qwen3-1.7B_no-thinking_100steps_4gpus`
- `tra_self_shot_Qwen3-1.7B_no-thinking_100steps_8gpus`
- `TRA_deepseek-7b_100steps_8gpus`

Qwen3、Qwen3.5、Gemma4 的当前配置均为 No thinking。若修改 `algorithm.enable_thinking: true`，SwanLab 名称会自动改为 `thinking`。DeepSeek Distill 使用原生推理模板，名称省略模式标签。

## 公共环境变量

先激活你的训练环境，再设置以下变量。将两处示例路径替换为服务器实际位置；凭据文件放在仓库外，文件内提供 `SWANLAB_API_KEY`。

```bash
export FRUGALOPL_PYTHON="$(command -v python)"
export FRUGALOPL_MODELS_ROOT="/path/to/models"
export FRUGALOPL_SWANLAB_ENV_FILE="/path/outside/repo/swanlab.env"
```

## Qwen3-1.7B · No thinking

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-1.7B"
# 4 卡 FKL
bash scripts/start_train.sh --no-tmux configs/training/fkl/qwen3_1p7b_100steps_4GPU.yaml
# 4 卡 Clip-OPSD
bash scripts/start_train.sh --no-tmux configs/training/clip_opsd/qwen3_1p7b_100steps_4GPU.yaml
# 8 卡 tra_self_shot（原 TRA）
bash scripts/start_train.sh --no-tmux configs/training/tra/qwen3_1p7b_100steps_8GPU.yaml
```

## Qwen3-4B · No thinking

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-4B"
# 4 卡 FKL
bash scripts/start_train.sh --no-tmux configs/training/fkl/qwen3_4b_100steps_4GPU.yaml
# 4 卡 Clip-OPSD
bash scripts/start_train.sh --no-tmux configs/training/clip_opsd/qwen3_4b_100steps_4GPU.yaml
# 8 卡 TRA
bash scripts/start_train.sh --no-tmux configs/training/tra/qwen3_4b_100steps_8GPU.yaml
```

## Qwen3.5-4B · No thinking

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3.5-4B"
# 4 卡 FKL
bash scripts/start_train.sh --no-tmux configs/training/fkl/qwen3_5_4b_100steps_4GPU.yaml
# 4 卡 Clip-OPSD
bash scripts/start_train.sh --no-tmux configs/training/clip_opsd/qwen3_5_4b_100steps_4GPU.yaml
# 8 卡 TRA
bash scripts/start_train.sh --no-tmux configs/training/tra/qwen3_5_4b_100steps_8GPU.yaml
```

## Qwen3.5-2B · No thinking

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3.5-2B"
# 4 卡 FKL
bash scripts/start_train.sh --no-tmux configs/training/fkl/qwen3_5_2b_100steps_4GPU.yaml
# 4 卡 Clip-OPSD
bash scripts/start_train.sh --no-tmux configs/training/clip_opsd/qwen3_5_2b_100steps_4GPU.yaml
# 8 卡 TRA
bash scripts/start_train.sh --no-tmux configs/training/tra/qwen3_5_2b_100steps_8GPU.yaml
```

## Gemma4-E2B-it · No thinking

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/gemma-4-E2B-it"
# 4 卡 FKL
bash scripts/start_train.sh --no-tmux configs/training/fkl/gemma4_e2b_it_100steps_4GPU.yaml
# 4 卡 Clip-OPSD
bash scripts/start_train.sh --no-tmux configs/training/clip_opsd/gemma4_e2b_it_100steps_4GPU.yaml
# 8 卡 TRA
bash scripts/start_train.sh --no-tmux configs/training/tra/gemma4_e2b_it_100steps_8GPU.yaml
```

## Gemma4-E4B-it · No thinking

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/gemma-4-E4B-it"
# 4 卡 FKL
bash scripts/start_train.sh --no-tmux configs/training/fkl/gemma4_e4b_it_100steps_4GPU.yaml
# 4 卡 Clip-OPSD
bash scripts/start_train.sh --no-tmux configs/training/clip_opsd/gemma4_e4b_it_100steps_4GPU.yaml
# 8 卡 TRA
bash scripts/start_train.sh --no-tmux configs/training/tra/gemma4_e4b_it_100steps_8GPU.yaml
```

## DeepSeek-R1-Distill-Qwen-1.5B

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B"
# 4 卡 FKL
bash scripts/start_train.sh --no-tmux configs/training/fkl/deepseek_1p5b_100steps_4GPU.yaml
# 4 卡 Clip-OPSD
bash scripts/start_train.sh --no-tmux configs/training/clip_opsd/deepseek_1p5b_100steps_4GPU.yaml
# 8 卡 TRA
bash scripts/start_train.sh --no-tmux configs/training/tra/deepseek_1p5b_100steps_8GPU.yaml
```

## DeepSeek-R1-Distill-Qwen-7B

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/DeepSeek-R1-Distill-Qwen-7B"
# 4 卡 FKL
bash scripts/start_train.sh --no-tmux configs/training/fkl/deepseek_7b_100steps_4GPU.yaml
# 4 卡 Clip-OPSD
bash scripts/start_train.sh --no-tmux configs/training/clip_opsd/deepseek_7b_100steps_4GPU.yaml
# 8 卡 TRA
bash scripts/start_train.sh --no-tmux configs/training/tra/deepseek_7b_100steps_8GPU.yaml
```

## TRA 指数参数 n

目标分布为

\[
r_i=\frac{p_i(p_i/q_i)^n\,\mathbf 1[p_i\ge f p_{\max}]}{\sum_j p_j(p_j/q_j)^n\,\mathbf 1[p_j\ge f p_{\max}]},
\qquad L=\mathrm{KL}(\operatorname{stopgrad}(r)\Vert p).
\]

参数在 `configs/common/tra_8GPU.yaml` 中定义，全部八份 TRA 配置继承：

```yaml
algorithm:
  tra_ratio_power: 1.0          # n，有限实数；默认目标 p²/q
  tra_plausible_fraction: 0.01  # f，支持集阈值
```

若只调整一个模型，在对应 TRA 配置末尾覆盖 `algorithm.tra_ratio_power`，例如 `0.5` 或 `2.0`。云端配置记录该参数，每次更新还在对应算法命名空间下记录 `tra_ratio_power`。检查点保存指数，恢复时核对一致性；旧的固定 n=1 TRA 检查点可以按默认值恢复。

## Qwen3-1.7B：三种 TRA 特权

三者共用同一个目标与超参数。现有 Qwen3-1.7B TRA 配置已明确命名为 `tra_self_shot`，即同题 solution 的 one-shot；上方该模型的八卡 TRA 命令即启动这一组。原 `TRAPlugin` 入口继续兼容。

`tra_random_shot` 对每道当前训练题，从完整配置训练集均匀、有放回地抽取一个题目及其 solution，作为 Teacher 的 one-shot。允许抽到同题。同一道题的全部回复共用该示例；使用 seed、batch ID 和当前题目确定抽样，跨 Worker、分片及恢复训练保持一致。示例池跟随 `controller.data.train`，无需另外配置路径。

`tra_true_answer` 只给 Teacher 当前题目与正确 `answer`，不插入示例，也不读取当前题目的 solution。Teacher 的单条 user 提示会说明正确答案，并要求依据该答案展开推理。三组 Student 提示完全相同。

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-1.7B"
# 8 卡：随机训练题目的 one-shot
bash scripts/start_train.sh --no-tmux configs/training/tra_random_shot/qwen3_1p7b_100steps_8GPU.yaml
# 8 卡：当前题目的正确答案
bash scripts/start_train.sh --no-tmux configs/training/tra_true_answer/qwen3_1p7b_100steps_8GPU.yaml
```

三组名称分别以 `tra_self_shot`、`tra_random_shot`、`tra_true_answer` 开头，仍含模型、`no-thinking`、步数和 GPU 数，group 仍为 `Qwen3-1.7B`。指数指标位于各自算法命名空间下，例如 `tra_random_shot/tra_ratio_power`。随机示例池的内容指纹和 seed 保存在本地检查点，用于恢复时核对；示例内容与路径不会上传 SwanLab。

## 双卡验证命令：必须有 tmux

以下轻量测试配置只保留在本地 `tests/`，不上传 GitHub。双卡验证使用 2 题 × 4 回复、2 次更新，训练生成上限 256、评测生成上限 128。

```bash
# 默认 n=1
bash scripts/start_train.sh --tmux tests/configs/swanlab_privacy/qwen3_1p7b.yaml
# n=2
bash scripts/start_train.sh --tmux tests/configs/swanlab_privacy/qwen3_1p7b_n2.yaml
```

正式 4/8 卡服务器使用上方的 `--no-tmux` 命令。基础八模型 × 三算法的双卡验证与吞吐见 [配置和验证总表](teacher_alignment_matrix.md)。

## 本次验证（2026-11-25）

CPU 检查 **166 项通过**，包括全部配置和启动命令、隐私参数过滤、SDK 采集开关、不同指数的独立梯度与检查点检查；默认 n=1 与原目标逐位一致。

双 RTX 4090 分别以 **n=1、n=2** 完成 2 次实际更新及 AMC23/AIME24 双卡评测。196 个 LoRA 目标层均产生有限非零更新，检查点和 SwanLab 中的指数值一致。在线读取了两组云端配置、元数据、日志、各数据集截断率和指数指标，未发现测试注入的个人信息或本地路径。

真实 SDK 的离线、在线异常退出检查也通过：终端中的模拟路径/邮箱不上传，自动异常钩子仅发送固定错误说明。该诊断会刻意生成一条 **0 steps / 0 GPUs、CRASHED** 的验证记录。

此前包含旧路径的 24 条 TCPO 云端测试记录已按用户要求删除，本地历史结果保留。本次完整测试证据位于本地 `tests/outputs/swanlab_privacy_20261125/`，临时检查点已清理。

## TRA 特权对照验证（2026-11-25）

三种特权方式的 CPU 检查 **149 项通过**，包括配置协议一致、随机示例跨 Worker/分片/恢复一致、提示长度处理、只给答案时无需 solution，以及实际 Learner 梯度更新。

使用原生 Qwen3 聊天模板和完整 **29,434 题**训练集核对了上下文：三组 Student 提示逐位相同；self/random 的 Teacher 为 user → assistant → user，true-answer 的 Teacher 为单条 user。

三组各在双 RTX 4090 上完成 **2 题 × 4 回复、2 次实际更新**，并完成两轮 AMC23/AIME24 双卡评测。CUDA Graph、196 个 LoRA 层的更新、检查点中的模式、评测期间预取和放行屏障、SwanLab 指标及隐私检查均通过。这是生成上限 256/128 的代码链路测试，正式八卡长序列训练由用户部署。

```bash
bash scripts/start_train.sh --tmux tests/configs/tra_privileges/tra_self_shot.yaml
bash scripts/start_train.sh --tmux tests/configs/tra_privileges/tra_random_shot.yaml
bash scripts/start_train.sh --tmux tests/configs/tra_privileges/tra_true_answer.yaml
```

本次测试证据位于本地 `tests/outputs/tra_privileges_20261125/`；测试配置和结果不上传 GitHub，临时检查点已清理，基础模型保留。
