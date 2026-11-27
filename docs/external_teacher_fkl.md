# 外部教师 Forward KL：编号 38、39

学生为 **Qwen3-1.7B**，独立冻结教师为 **Qwen3-4B-Instruct-2507**，双方均 No thinking。教师与学生部署在每个 Learner 的同一张 GPU 上，八卡均参与学生 rollout、学习和评测，只更新学生 LoRA。

| 编号 | 实验 | 教师上下文 | 配置 | 正式状态 |
| --- | --- | --- | --- | --- |
| 38 | FKL_external | 仅当前题目，无特权 | [八卡配置](../configs/training/external_fkl/qwen3_1p7b_teacher_qwen3_4b_2507_100steps_8GPU.yaml) | 待运行 |
| 39 | OPD_external_self_shot | 当前题目 → 正确 solution → 当前题目，conversation one-shot | [八卡配置](../configs/training/external_opd/qwen3_1p7b_teacher_qwen3_4b_2507_100steps_8GPU.yaml) | 待运行 |

## 目标与上下文

学生从不接收特权。对学生 on-policy 回复的每个位置，由独立教师评分，最小化 `KL(q_teacher || p_student)`，不使用 CLIP、奖励权重或 RL 损失。沿用已有 FKL 的 Student top-512 支持集，双方在该支持集内归一化；按整个 batch 的有效回复 token 统一归约。

教师加载自己的 tokenizer、chat template、backbone 和输出头，并设置 `enable_thinking=false`。启动前严格核对两个模型的 token→ID 词表、特殊 token 与输出词表维度，再使用同一组回复 token 对齐评分。两个模型的隐藏维度分别为 2048／2560，分别使用各自输出头。教师保持 eval、参数冻结，不加入学生优化器，不随学生更新；检查点只保存学生 LoRA、优化器和算法状态。

编号 39 的教师上下文为 `user(当前题目) → assistant(正确 solution) → user(当前题目)`，保留原有同题 one-shot 方式。只有 solution 可按提示预算缩短，题目和 generation suffix 保留。编号 38 的教师仅接收当前题目。

## 统一参数

- 8 卡，每卡教师／学生共卡；训练 100 次更新，global batch 32 题×8 回复。
- 训练生成上限 8192，temperature=1.0、top-p=1.0；评测生成上限 32768，temperature=0.6、top-p=0.95。
- 初始及每 10 步依次在 AMC23、AIME24 评测 Avg@16；同题回复跨卡分散，每个数据集记录截断率。
- 每 10 步保存学生 LoRA；学习率 5e-6，LoRA r=64、alpha=128、dropout=0。
- vLLM 使用 CUDA Graph，`enforce_eager=false`。评测预取下一批 rollout 后，仍须等待评测全部结束才放行 Learner。
- SwanLab 为公开 TCPO Project，group=Qwen3-1.7B，只公开教师模型名称和算法超参，不上传本地模型路径或提示文本。

## 八卡正式启动：无 tmux

先激活环境、进入项目目录，修改路径；两组分别启动：

```bash
export FRUGALOPL_PYTHON="$(command -v python)"
export FRUGALOPL_MODEL_PATH="/path/to/models/Qwen3-1.7B"
export FRUGALOPL_TEACHER_MODEL_PATH="/path/to/models/Qwen3-4B-Instruct-2507"
export FRUGALOPL_SWANLAB_ENV_FILE="/path/outside/repo/swanlab.env"
# 编号 38：无特权
bash scripts/start_train.sh --no-tmux configs/training/external_fkl/qwen3_1p7b_teacher_qwen3_4b_2507_100steps_8GPU.yaml
```

```bash
export FRUGALOPL_PYTHON="$(command -v python)"
export FRUGALOPL_MODEL_PATH="/path/to/models/Qwen3-1.7B"
export FRUGALOPL_TEACHER_MODEL_PATH="/path/to/models/Qwen3-4B-Instruct-2507"
export FRUGALOPL_SWANLAB_ENV_FILE="/path/outside/repo/swanlab.env"
# 编号 39：教师接收同题 solution one-shot
bash scripts/start_train.sh --no-tmux configs/training/external_opd/qwen3_1p7b_teacher_qwen3_4b_2507_100steps_8GPU.yaml
```

两个模型必须使用独立路径。切回普通自蒸馏配置前，执行 `unset FRUGALOPL_TEACHER_MODEL_PATH`。

## 双卡验证：有 tmux

本地轻量配置位于忽略提交的 `tests/` 中，每组使用 2 题×4 回复、2 次实际更新，训练／评测生成上限 256／128，检查 CUDA Graph、共卡教师评分、LoRA 更新、短评测及预取屏障：

```bash
bash scripts/start_train.sh --tmux tests/configs/external_teacher/plain.yaml
bash scripts/start_train.sh --tmux tests/configs/external_teacher/privileged.yaml
```

正式八卡实验由用户部署，双卡短链路验证不计作正式实验完成。验证结果保存在本地 `tests/outputs/external_teacher_20261127/`。

2026-11-27 验证完成：227 项 WSL CPU 检查通过；两组分别在远程双 RTX 4090 完成 2 次实际更新及两轮 AMC23 → AIME24 短评测。每组审计 16 条 rollout，确认每卡教师／学生同设备、教师参数冻结且无梯度、独立输出头正确、196 个学生 LoRA 层均更新；无 CLIP loss、优化器状态、检查点模式、CUDA Graph、各数据集截断率及评测预取屏障均通过。没有新增云端测试 run，临时检查点已清理，GPU 已释放。
