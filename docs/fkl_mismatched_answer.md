# FKL_mismatched_answer

使用 Qwen3-1.7B No thinking。每条 rollout 各自接收一个冲突答案，Teacher 在同一个 user 消息中看到当前题目与该答案。Student 只看到原题。优化目标为无 CLIP 的 Forward KL，Student top-k 默认为 512；不增加奖励损失或其他权重。

## 答案选择

1. 从最终完整 `\boxed{...}` 提取每条 rollout 的答案。
2. 在同一道题的其他 rollout 中，保留答案可提取且与当前答案不同的候选。比较会处理常见数值、分数、选项内容、单位和等价表达，避免将 `C` 与其对应的数值当成冲突。
3. 从可用候选回复中确定性采样一个；种子由运行 seed、batch ID、题目及当前回复决定。样本分片和 Learner 领取顺序不改变所选答案。
4. 若当前回复未能提取答案，或同题没有不同答案，基于标准答案扰动数值或表达式，并再次排除与当前答案相同的结果。二元答案空间若没有第三个合法答案，而标准答案本身已与 rollout 不同，则使用该标准答案，并单独统计。
5. 在完整同题回复组上完成选择后，才将逐条样本交给异步 Learner。不会跨题借用答案，也不按回复正误切换损失。

缺失、null、空白标准答案仍在训练集加载时剔除，数值 0 保留。输入记录中的标准答案不被修改，评测仍使用真实标准答案。

## 反向对照：TCA_mismatch_answer

新增 Qwen3-1.7B 八卡配置 `configs/training/tra_mismatch_answer/qwen3_1p7b_100steps_8GPU.yaml`。它与本实验共用 `MismatchedAnswerTeacherMixin` 的逐 rollout 答案选择、单轮 Teacher 提示及回退规则，区别在于拟合 `r ∝ p × (p/q)^n`，默认 `n=1`、支持集阈值 `0.01`。仓库入口仍使用 TRA 名称，对应本轮清单中的 TCA。FKL 原有检查点字段与答案选择版本保持兼容。

```bash
export FRUGALOPL_MODEL_PATH="/path/to/models/Qwen3-1.7B"
bash scripts/start_train.sh --no-tmux configs/training/tra_mismatch_answer/qwen3_1p7b_100steps_8GPU.yaml
```

## 八卡正式启动，无 tmux

激活训练环境，在项目根目录执行，并替换实际路径：

```bash
export FRUGALOPL_PYTHON="$(command -v python)"
export FRUGALOPL_MODEL_PATH="/path/to/models/Qwen3-1.7B"
export FRUGALOPL_SWANLAB_ENV_FILE="/path/outside/repo/swanlab.env"
bash scripts/start_train.sh --no-tmux configs/training/fkl_mismatched_answer/qwen3_1p7b_100steps_8GPU.yaml
```

全局 batch 为 32 题 × 每题 8 回复，训练 100 steps；LoRA r=64、alpha=128，学习率 5e-6。训练最大回复 8192，temperature=1、top-p=1；评测最大回复 32768，temperature=0.6、top-p=0.95。初始及每 10 steps 依次评测 AMC23、AIME24，Avg@16，并按既有配置保存检查点。

Actor 与 Learner 共用八卡，维持现有异步流水线及评测期间的 rollout 预取屏障。评测的同题回复分散至多卡，逐数据集记录截断率。SwanLab 使用公开的 TCPO Project，group 为 Qwen3-1.7B，上传配置使用既有隐私过滤。

## 双卡验证，有 tmux

以下配置、夹具和输出均仅在本地 `tests/` 中保留，不提交 GitHub：

```bash
bash scripts/start_train.sh --tmux tests/configs/fkl_mismatched_answer/dual_gpu.yaml
```

真实生成测试使用 2 题 × 4 回复、2 次更新。另用预先保存的真实多样化回复做单独标记的双卡回放，覆盖同题候选选择分支、缺失答案回退及真实模型前反向传播。回放结果只用于代码链路验证，不作为性能实验。

答案特权默认单轮同样应用于 `FKL_true_answer`。解题 solution 的 one-shot 实验沿用原有定义；历史答案 one-shot 的检查点与新的单轮目标不兼容。

验证结果：113 项 CPU 检查通过；双卡真实生成与单独标记的回放各完成 2 次优化，196 个 LoRA 层均有有限、非零更新。逐样本审计覆盖 12 条同题候选选择和 4 条回退选择，全部与当前可提取答案不一致，Teacher 均为单轮。CUDA Graph、AMC23/AIME24 双卡评测、评测预取屏障及 SwanLab 隐私检查均通过；测试检查点已删除，双卡已释放。八卡配置由用户在正式服务器启动，以上为双卡验证结果。
