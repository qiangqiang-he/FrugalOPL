# 当前三模型 × 三算法及 TRA 特权对照：启动命令

当前启动优先级以 [实验优先级与启动清单](实验优先级与启动清单.md) 为准：第一阶段仅运行 Qwen3-1.7B、Qwen3-4B、Qwen3-8B 的九组主实验，新增的 8B 和答案不匹配配置也在该清单中。以下仅列出这三个模型的启动配置。

分支：`TCPO_v1`。FKL、Clip-OPSD 的 Qwen3-4B、Qwen3-8B 使用 **8 卡**、Qwen3-1.7B 使用 **4 卡**；基础 Teacher reverse align（TRA）使用 **8 卡**。Qwen3-1.7B 的 n=0.2 指数对照使用 **4 卡**；所有 n=0.05、Qwen3-1.7B 的 n=0.1／0.3 以及 Qwen3-4B、8B 的 n=0.1／0.2 使用 **8 卡**。新增三组 TRA alpha=0.5、n=0.1 也均为 **8 卡**。正式服务器均通过 `--no-tmux` 在前台启动。以下包含 9 个基础实验、4 个 TRA 特权对照、10 个指数对照、3 个损失系数对照、2 个四卡空答案对照、3 个 standard_OPSD（1.7B 四卡、4B／8B 八卡）、3 个八卡 standard_FKL，以及 2 个八卡外部 Qwen3-4B-Instruct-2507 教师对照（编号 38、39），以及 3 个八卡 Thinking TRA 对照（编号 40–42）。

所有配置默认训练 100 次更新，全局 batch 为 32 题 × 每题 8 个回复，训练生成上限 8192；初始及每 10 步依次评测 AMC23、AIME24，Avg@16，生成上限 32768。LoRA r=64、alpha=128，分布损失使用 Student top-512。

## SwanLab 记录范围与名称

云端只上传选定的实验参数和数值指标：模型公开名称、算法、thinking 模式、采样与优化参数、LoRA 参数、准确率、长度、熵、各数据集截断率及吞吐。配置中的本地模型路径、数据/输出路径、Redis 地址、凭据、提示词内容均不会复制到云端。

SDK 的终端采集、运行环境、主机信息、Git、依赖快照和 SwanLab 本地目录采集关闭；数值型 GPU/CPU 利用率监控保留。异常详情保留在本地日志，云端只记录固定说明。此边界通过 SDK 的 `Settings` 配置，在初始化前生效，参考 [官方设置说明](https://docs.swanlab.cn/api/py-settings.html)。本地 YAML 和启动日志仍保留完整内容，便于调试。

项目为公开的 [qhe/TCPO](https://swanlab.cn/@qhe/TCPO)，group 使用模型名称。名称示例：

- `FKL_Qwen3-1.7B_no-thinking_100steps_4gpus`
- `tra_self_shot_Qwen3-1.7B_no-thinking_100steps_8gpus`
- `TRA_Qwen3-8B_no-thinking_100steps_8gpus`

基础配置为 No thinking；新增编号 40–42 明确启用 Thinking，学生、特权教师与评测共用 `algorithm.enable_thinking: true`，SwanLab run 名含 `thinking`。

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
# 8 卡 FKL
bash scripts/start_train.sh --no-tmux configs/training/fkl/qwen3_4b_100steps_8GPU.yaml
# 8 卡 Clip-OPSD
bash scripts/start_train.sh --no-tmux configs/training/clip_opsd/qwen3_4b_100steps_8GPU.yaml
# 8 卡 TRA
bash scripts/start_train.sh --no-tmux configs/training/tra/qwen3_4b_100steps_8GPU.yaml
```

## Qwen3-8B · No thinking

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-8B"
# 8 卡 FKL
bash scripts/start_train.sh --no-tmux configs/training/fkl/qwen3_8b_100steps_8GPU.yaml
# 8 卡 Clip-OPSD
bash scripts/start_train.sh --no-tmux configs/training/clip_opsd/qwen3_8b_100steps_8GPU.yaml
# 8 卡 TRA
bash scripts/start_train.sh --no-tmux configs/training/tra/qwen3_8b_100steps_8GPU.yaml
```

## TRA 指数参数 n

目标分布为

\[
r_i=\frac{p_i(p_i/q_i)^n\,\mathbf 1[p_i\ge f p_{\max}]}{\sum_j p_j(p_j/q_j)^n\,\mathbf 1[p_j\ge f p_{\max}]},
\qquad L=\alpha\,\mathrm{KL}(\operatorname{stopgrad}(r)\Vert p).
\]

参数在 `configs/common/tra_8GPU.yaml` 中定义，当前 TRA 配置继承：

```yaml
algorithm:
  tra_ratio_power: 1.0          # n，有限实数；默认目标 p²/q
  tra_loss_alpha: 1.0           # alpha，范围 [0,1]，整体缩放 KL loss
  tra_plausible_fraction: 0.01  # f，支持集阈值
```

若只调整一个模型，在对应 TRA 配置末尾覆盖 `algorithm.tra_ratio_power`，例如 `0.5` 或 `2.0`。云端配置记录该参数，每次更新还在对应算法命名空间下记录 `tra_ratio_power`。检查点保存指数，恢复时核对一致性；旧的固定 n=1 TRA 检查点可以按默认值恢复。

损失系数使用 `algorithm.tra_loss_alpha`，默认 1 保持原损失和梯度；每步记录 `tra_raw_loss`、`tra_loss` 和 `tra_loss_alpha`。旧检查点未记录 alpha 时按 1 兼容恢复；alpha=0 时跳过优化器更新。详见 [TRA 损失系数](tra_loss_alpha.md)。

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

## Qwen3-4B、Qwen3-8B：TRA 不匹配 solution one-shot

两组同题 solution TRA 及各自的 Clip-OPSD 均已由用户确认完成。新增 TRA 对照从训练集另一道题抽取完整的“题目＋solution”作为 one-shot，排除当前题的全部同名记录。同题回复共用示例；Student 提示不含示例。训练及评测协议与原 TRA 相同，使用 8 卡、100 steps、batch 32×8、训练 8192／评测 32768、Avg@16。

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-4B"
bash scripts/start_train.sh --no-tmux configs/training/tra_mismatch_solution/qwen3_4b_100steps_8GPU.yaml
```

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-8B"
bash scripts/start_train.sh --no-tmux configs/training/tra_mismatch_solution/qwen3_8b_100steps_8GPU.yaml
```

与既有 `tra_random_shot` 区别：新配置严格排除同题，使用 `TRAMismatchedSolutionPlugin`；原随机示例配置保留既有抽样行为。远程双卡暂不可用，本次使用 CPU 检查配置、上下文和优化链路，未作 GPU 实测。

## 三个模型：同题 solution TRA，n=0.2

三组将 TRA 指数改为 `algorithm.tra_ratio_power: 0.2`，特权仍为同题 solution one-shot。Qwen3-1.7B 使用四卡，新参数标为待重启；Qwen3-4B、Qwen3-8B 使用八卡、待运行。均为 100 steps、batch 32×8，训练 8192／评测 32768、Avg@16；SwanLab run name 用 `n0p2` 区分。

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-1.7B"
bash scripts/start_train.sh --no-tmux configs/training/tra/qwen3_1p7b_n0p2_100steps_4GPU.yaml
```

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-4B"
bash scripts/start_train.sh --no-tmux configs/training/tra/qwen3_4b_n0p2_100steps_8GPU.yaml
```

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-8B"
bash scripts/start_train.sh --no-tmux configs/training/tra/qwen3_8b_n0p2_100steps_8GPU.yaml
```

## Qwen3-1.7B：四卡 match／mismatch 空答案对照（20–21）

两组均为 No thinking、四卡、n=1、100 steps、batch 16×8，训练上限 8192／评测上限 32768、Avg@16。Teacher 使用单轮答案提示；空答案回复继续参与优化。

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-1.7B"
# 编号 20：当前 rollout 自身答案；提取失败用空字符串
bash scripts/start_train.sh --no-tmux configs/training/tra_match_answer/qwen3_1p7b_empty_fallback_100steps_4GPU.yaml
```

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-1.7B"
# 编号 21：同题不匹配答案；没有可用候选或提取失败用空字符串
bash scripts/start_train.sh --no-tmux configs/training/tra_mismatch_answer/qwen3_1p7b_empty_fallback_100steps_4GPU.yaml
```

## Qwen3-1.7B：八卡同题 solution TRA，n=0.1／0.3（22–23）

两组分别设置 `algorithm.tra_ratio_power: 0.1`、`0.3`，使用八卡；其余训练参数沿用编号 17：No thinking、100 steps、batch 32×8、训练 8192／评测 32768、AMC23 和 AIME24 Avg@16。特权均为同题 solution one-shot。

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-1.7B"
# 编号 22：n=0.1
bash scripts/start_train.sh --no-tmux configs/training/tra/qwen3_1p7b_n0p1_100steps_8GPU.yaml
```

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-1.7B"
# 编号 23：n=0.3
bash scripts/start_train.sh --no-tmux configs/training/tra/qwen3_1p7b_n0p3_100steps_8GPU.yaml
```

## standard_OPSD：单轮 solution 特权＋CLIP，四卡／八卡（24–26）

Teacher 采用官方 OPSD `reason_first=False` 的单轮拼接方式，当前题目只出现一次，额外特权为 solution。Qwen3-1.7B 使用四卡、CLIP epsilon=1e-6；Qwen3-4B 使用八卡、CLIP epsilon=1e-6；Qwen3-8B 使用八卡、CLIP epsilon=1e-7。保留 Student top-512；三组均为 No thinking、100 steps、batch 32×8、训练 8192／评测 32768、AMC23 和 AIME24 Avg@16。详见 [实现与验证说明](standard_opsd.md)。

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-1.7B"
# 编号 24
bash scripts/start_train.sh --no-tmux configs/training/standard_opsd/qwen3_1p7b_100steps_4GPU.yaml
```

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-4B"
# 编号 25
bash scripts/start_train.sh --no-tmux configs/training/standard_opsd/qwen3_4b_100steps_8GPU.yaml
```

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-8B"
# 编号 26
bash scripts/start_train.sh --no-tmux configs/training/standard_opsd/qwen3_8b_100steps_8GPU.yaml
```

## standard_FKL：标准 OPSD 形式的单轮 solution 特权，无 CLIP（27–29）

Teacher 提示与 standard_OPSD 完全相同，按“当前题目一次＋solution＋transition”拼接；`kl_token_clip: null`，使用普通 Forward KL。三组均为八卡、No thinking、100 steps、batch 32×8、训练 8192／评测 32768、AMC23 和 AIME24 Avg@16。详见 [standard_FKL](standard_fkl.md)。

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-1.7B"
# 编号 27
bash scripts/start_train.sh --no-tmux configs/training/standard_fkl/qwen3_1p7b_100steps_8GPU.yaml
```

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-4B"
# 编号 28
bash scripts/start_train.sh --no-tmux configs/training/standard_fkl/qwen3_4b_100steps_8GPU.yaml
```

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-8B"
# 编号 29
bash scripts/start_train.sh --no-tmux configs/training/standard_fkl/qwen3_8b_100steps_8GPU.yaml
```

## Qwen3-4B／8B：八卡同题 solution TRA，n=0.1（30–31）

n=0.2 已有八卡配置，对应编号 18、19。下面仅新增 n=0.1 两组，训练及评测设置沿用同模型的 n=0.2：No thinking、100 steps、batch 32×8、训练 8192／评测 32768、AMC23 和 AIME24 Avg@16，特权为同题 solution one-shot。

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-4B"
# 编号 30：n=0.1
bash scripts/start_train.sh --no-tmux configs/training/tra/qwen3_4b_n0p1_100steps_8GPU.yaml
```

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-8B"
# 编号 31：n=0.1
bash scripts/start_train.sh --no-tmux configs/training/tra/qwen3_8b_n0p1_100steps_8GPU.yaml
```

## 三个模型：八卡同题 solution TRA，n=0.05（32–34）

三组均设置 `algorithm.tra_ratio_power: 0.05`，使用八卡，特权为同题 solution one-shot。沿用 No thinking、100 steps、batch 32×8、训练 8192／评测 32768、AMC23 和 AIME24 Avg@16；初始及每 10 步评测，并每 10 步保存 LoRA。

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-1.7B"
# 编号 32：n=0.05
bash scripts/start_train.sh --no-tmux configs/training/tra/qwen3_1p7b_n0p05_100steps_8GPU.yaml
```

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-4B"
# 编号 33：n=0.05
bash scripts/start_train.sh --no-tmux configs/training/tra/qwen3_4b_n0p05_100steps_8GPU.yaml
```

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-8B"
# 编号 34：n=0.05
bash scripts/start_train.sh --no-tmux configs/training/tra/qwen3_8b_n0p05_100steps_8GPU.yaml
```

## 三个模型：八卡 TRA alpha=0.5，n=0.1（35–37）

沿用同题 solution one-shot、No thinking、100 steps、batch 32×8、训练 8192／评测 32768、AMC23 和 AIME24 Avg@16；只将 TRA loss 整体乘以 alpha=0.5，n 仍为 0.1。配置与测试说明见 [损失系数文档](tra_loss_alpha.md)。

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-1.7B"
# 编号 35
bash scripts/start_train.sh --no-tmux configs/training/tra/qwen3_1p7b_n0p1_alpha0p5_100steps_8GPU.yaml
```

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-4B"
# 编号 36
bash scripts/start_train.sh --no-tmux configs/training/tra/qwen3_4b_n0p1_alpha0p5_100steps_8GPU.yaml
```

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-8B"
# 编号 37
bash scripts/start_train.sh --no-tmux configs/training/tra/qwen3_8b_n0p1_alpha0p5_100steps_8GPU.yaml
```

## 外部 4B 教师 → 1.7B 学生，八卡共卡（38–39）

双方 No thinking，教师冻结，Student top-512、无 CLIP Forward KL；100 steps，batch 32×8，训练 8192／AMC23、AIME24 评测 32768 Avg@16。两组独立启动。每个 GPU 都部署教师和学生，只有学生 LoRA 更新。详见 [说明与验证](external_teacher_fkl.md)。

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-1.7B"
export FRUGALOPL_TEACHER_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-4B-Instruct-2507"
# 编号 38：无特权，双方只看题目
bash scripts/start_train.sh --no-tmux configs/training/external_fkl/qwen3_1p7b_teacher_qwen3_4b_2507_100steps_8GPU.yaml
```

```bash
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-1.7B"
export FRUGALOPL_TEACHER_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-4B-Instruct-2507"
# 编号 39：教师额外接受同题 solution one-shot
bash scripts/start_train.sh --no-tmux configs/training/external_opd/qwen3_1p7b_teacher_qwen3_4b_2507_100steps_8GPU.yaml
```

切回普通自蒸馏配置前执行 `unset FRUGALOPL_TEACHER_MODEL_PATH`，避免误用外部教师路径。

## 三个模型：八卡 Thinking TRA，alpha=0.1、n=0.1（40–42）

均为同题 solution one-shot，学生／教师／评测全部 Thinking。设置 `tra_loss_alpha=0.1`、`clip_grad_norm=0.05`，n 继续为 0.1；LoRA alpha=128。沿用 100 steps、batch 32×8、训练 8192／评测 32768、初始及每 10 步 AMC23、AIME24 Avg@16，其他参数保持不变。

```bash
unset FRUGALOPL_TEACHER_MODEL_PATH
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-1.7B"
# 编号 40
bash scripts/start_train.sh --no-tmux configs/training/tra/qwen3_1p7b_thinking_n0p1_alpha0p1_100steps_8GPU.yaml
```

```bash
unset FRUGALOPL_TEACHER_MODEL_PATH
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-4B"
# 编号 41
bash scripts/start_train.sh --no-tmux configs/training/tra/qwen3_4b_thinking_n0p1_alpha0p1_100steps_8GPU.yaml
```

```bash
unset FRUGALOPL_TEACHER_MODEL_PATH
export FRUGALOPL_MODEL_PATH="${FRUGALOPL_MODELS_ROOT}/Qwen3-8B"
# 编号 42
bash scripts/start_train.sh --no-tmux configs/training/tra/qwen3_8b_thinking_n0p1_alpha0p1_100steps_8GPU.yaml
```

本地双卡验证使用 tmux 和轻量配置：`bash scripts/start_train.sh --tmux tests/configs/tra_thinking/qwen3_1p7b.yaml`。测试配置与结果只保留在本地，不提交 GitHub。

145 项 WSL CPU 检查通过；Qwen3-1.7B 完成双卡 2 次实际更新及短评测，确认训练学生、特权教师和评测均为原生 Thinking 模板，loss 缩放为 0.1 倍，196 个 LoRA 层均更新。4B、8B 完成配置和 CPU 链路检查；正式八卡由用户部署。

## 双卡验证命令：必须有 tmux

以下轻量测试配置只保留在本地 `tests/`，不上传 GitHub。双卡验证使用 2 题 × 4 回复、2 次更新，训练生成上限 256、评测生成上限 128。

```bash
# 默认 n=1
bash scripts/start_train.sh --tmux tests/configs/swanlab_privacy/qwen3_1p7b.yaml
# n=2
bash scripts/start_train.sh --tmux tests/configs/swanlab_privacy/qwen3_1p7b_n2.yaml
```

正式 4/8 卡服务器使用上方的 `--no-tmux` 命令。当前模型的配置和验证记录见 [配置和验证总表](teacher_alignment_matrix.md)。

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
