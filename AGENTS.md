# 项目工作规范

## 项目目标

本项目用于开发、测试并验证强化学习（RL）算法。正式算法训练最终在 8 卡服务器上运行，算法开发过程中的测试与验证默认在本地完成。

## 代码开发与运行环境

- 代码编写在本地（Windows）当前工作区完成；所有代码的**运行与调试**一律在 WSL 的 Ubuntu 中进行。
- 所有会用到的模型统一存放在 WSL 的 `\\wsl.localhost\Ubuntu\home\qqh\projects\models` 路径下，引用模型时一律使用该目录，不得另行下载或复制模型。
- 代码在当前工作区写好后，须将项目同步到 WSL 的 `\\wsl.localhost\Ubuntu\home\qqh\projects\` 下同名目录再运行。例如当前项目名为 `AAA_Project`，则同步到 `\\wsl.localhost\Ubuntu\home\qqh\projects\AAA_Project`，最终的运行与调试都在该目录中进行。
- WSL 内对应的模型路径为 `~/projects/models`，项目路径为 `~/projects/<项目名>`。
- 在 WSL 中运行、测试、训练或评估产生的所有结果文件，必须同步回 Windows 当前工作区的对应目录；包括输出、日志、检查点和测试结果，确保 Windows 工作区始终保留最新结果。

## 本地测试环境

- 默认在 WSL 中执行所有代码测试，使用 Conda 环境 `py312-th210-vl019-cu128`，通过 `conda activate py312-th210-vl019-cu128` 激活。
- 测试应同时覆盖必要的 CPU 测试和可行的单卡 GPU 测试。

## SwanLab 凭据

- OPSD、OPSD 的所有变体以及 TCDO 的训练、评估和相关诊断，统一记录到 SwanLab 的 `TCDO` Project。
- 需要记录 SwanLab 的 WSL 任务，应在启动前加载仓库外的
  `/home/qqh/.config/frugalopl/swanlab.env`；该文件提供
  `SWANLAB_API_KEY`，目录权限应为 `700`、文件权限应为 `600`。
- 不得将 SwanLab API key 写入 `AGENTS.md`、仓库配置、源码、测试产物、
  日志或 Git 提交。

## 显存策略（强制要求）

- **本地 Windows/WSL 开发与测试**必须保留 **3 GB 显存安全线**：任何时刻显存占用不得超过 `总显存 − 3 GB`，避免本机完全卡死。配置 batch size、rollout 数量、并发数或加载模型前，均须先核对该安全线。
- **远程服务器正式训练**不设置固定的空闲显存安全线，也不得因预检中的固定保留显存而拒绝已配置的训练。服务器的目标是在不浪费可用显存的前提下，通过实测显存、合理的并发/批次配置、Worker 交接时释放可回收 CUDA 缓存，尽量避免真实 CUDA OOM 导致训练中止。
- 服务器上仍须在启动前核对模型常驻显存与预期峰值；发现真实 OOM 风险时，应调整配置或显存交接策略，而不是人为保留本地的 3 GB 阈值。

## 禁止开启 Eager（强制要求）

- 训练、评估和测试均禁止开启 Eager 模式；vLLM 必须使用 CUDA Graph，`enforce_eager` 必须为 `false`。
- 不得以显存安全、避免 OOM、兼容性检查或调试为由开启 Eager，也不得在出错后自动回退到 Eager。
- 显存不足时应通过调整显存预算、并发、批次或 Worker 显存交接策略解决；本地仍须保留上述 3 GB 安全线，且不得关闭 CUDA Graph。

## 测试文件管理

- 所有测试代码、测试脚本和测试配置必须放在 `tests/` 目录中。
- 所有测试产生的输出、日志、检查点和临时 LoRA 也必须放在 `tests/` 内；统一使用 `tests/outputs/`，启动日志统一使用 `tests/outputs/.logs/`，不得写入顶层 `outputs/`。
- 训练测试配置必须显式设置 `run.storage.output_root: tests/outputs`；评估测试配置必须将 `run.output_dir` 设置在 `tests/outputs/` 下。
- `tests/` 目录中的新增或修改仅供本地验证，不得提交或推送到 GitHub。

## 训练与评估启动规范

- 训练和评估一律通过“启动脚本 + 配置文件”启动；不得直接运行底层 Python 入口。
- 双卡 4090 的训练命令为：

  ```bash
  bash scripts/start_train.sh --tmux configs/xxxx.yaml
  ```

- 双卡 4090 的评估命令为：

  ```bash
  bash scripts/start_eval.sh --tmux configs/xxxx.yaml
  ```

- 每次运行由 `configs/` 下的 YAML 配置文件完整定义；正式训练配置默认面向 8 卡服务器。
- **双卡 4090 的训练、评估和 GPU 调试必须使用 tmux 托管**，避免终端、SSH 或 Codex 会话断开导致任务中止。两个启动脚本均默认使用 tmux，也可显式传入 `--tmux`；不得因为 8 卡使用前台启动而取消双卡的 tmux 保护。
- **8 卡服务器由用户操作，Codex 无法直接连接。8 卡服务器的训练和评估使用 shell 脚本加 `--no-tmux` 启动**，脚本直接在前台 `exec` Python，不创建 tmux 会话、不自行转为后台任务。命令分别为 `bash scripts/start_train.sh --no-tmux configs/xxxx.yaml` 和 `bash scripts/start_eval.sh --no-tmux configs/xxxx.yaml`。
- 编写或更新 8 卡启动方案时，交付说明必须明确标注无 tmux 的 8 卡命令，以及有 tmux 的双卡验证命令。先用双卡和小模型验证同一套算法与训练流程，再交付 8 卡配置；不能把双卡通过描述成已经在 8 卡上实测。
- 如需在双卡验证 `--no-tmux` 前台入口本身，必须先由外层 tmux 托管该验证任务；入口脚本仍直接执行 Python，不再创建内层 tmux。普通双卡训练和评估继续使用默认模式或显式 `--tmux`。

## 本地轻量训练测试

- **双卡 4090 默认全局训练 batch 为 32 个问题 × 每题 4 个回复，共 128 条 rollout**；32 是整个 batch 的问题总数。用户明确要求的小规模链路验证可缩小 batch，正式双卡实验按此默认值配置。
- 测试算法时，可在 `tests/configs/` 中创建轻量配置，使用更短的生成长度、更小的模型和更小的 batch size。
- 本地单卡训练统一使用 GSM8K：训练集为 `tests/data/gsm8k_train.json`，评测集为 `tests/data/gsm8k_test.json`。`tests/configs/` 中的单卡训练配置必须分别使用这两个文件作为训练集和评测集。
- 本地单卡训练的最大生成长度统一设置为 `1024`，即 `data.max_response_length: 1024`。
- 推荐的最小训练规模为 `2 × 4`：2 个 question，每个 question 生成 4 个 rollout，并训练 2 个 batch。
- 本地 GPU 的 rollout 速度较快。使用 vLLM 时，通常 3 分钟可完成约 30–40 个 rollout；若 8 个 rollout 超过 3 分钟仍未完成，应停止等待并检查配置、进程或运行状态。

## 本地评估规范

- 评估时单次 rollouts 数量统一控制为 **64**，该值为效率最大化的 rollout 数量，不得随意调大。

## 本文件的修改权限

- 未经用户明确要求，不得修改本文件。
- 如确有必要修改，必须先向用户说明原因并获得明确许可。
