# TCA 模型代码兼容性检查

2026-11-25，四个模型先通过 WSL 的 20 项 CPU/静态检查，随后均在远程双 RTX 4090 完成轻量 TCA 实训及最终检查。每组加载完整预训练权重，以 2 个问题 × 每题 4 个回复完成 2 次实际优化更新；两卡均同时部署 Actor 和 Learner，使用 tmux 托管，vLLM 全程开启 CUDA Graph。

| 模型 | 原生架构 | 实际更新的 LoRA 模块数 | 配置 |
| --- | --- | ---: | --- |
| Qwen/Qwen3-4B | Qwen3ForCausalLM | 252 | `configs/training/tca/qwen3_4b_100steps_8GPU.yaml` |
| Qwen/Qwen3.5-4B | Qwen3_5ForConditionalGeneration | 248 | `configs/training/tca/qwen3_5_4b_100steps_8GPU.yaml` |
| google/gemma-4-E4B-it | Gemma4ForConditionalGeneration | 258 | `configs/training/tca/gemma4_e4b_it_100steps_8GPU.yaml` |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | Qwen2ForCausalLM | 196 | `configs/training/tca/deepseek_r1_distill_qwen_1p5b_100steps_8GPU.yaml` |

GPU 实测检查了以下行为：

- Student 生成、特权 Teacher 前向、top-k=512 TCA 反向及优化器更新；两步梯度范数均有限且非零。
- 最终适配器中全部 A/B 维度与真实基础权重逐层对应，全部 B 矩阵有限且非零，每个模块的 LoRA 增量探针均非零，优化器状态有限。
- 更新后的 step 1/2 LoRA 均由 Actor 成功加载，随后完成生成。
- 每步按 AMC23 → AIME24 评测，同题回复分散到两卡；评测末尾预取下一批训练数据、共享 Actor 任务池及 Learner 交接正常。两个数据集的截断率均上传 SwanLab。

这是小规模执行检查：训练生成上限 Qwen3/Qwen3.5/Gemma 为 128，DeepSeek 为 256；评测每数据集仅 1 题、Avg@2、生成上限 64。上述评测结果用于检查流程与指标记录。正式 8 卡配置仍保留原有 batch、生成长度及 Avg@16，尚未在 8 卡上实测。

SwanLab 最终通过记录均在 `TCDO` Project：[Qwen3-4B](https://swanlab.cn/@qhe/TCDO/runs/f1xplsdy/chart)、[Qwen3.5-4B](https://swanlab.cn/@qhe/TCDO/runs/us9ezoij/chart)、[Gemma-4-E4B-it](https://swanlab.cn/@qhe/TCDO/runs/gfuqlvcc/chart)、[DeepSeek Distill 1.5B](https://swanlab.cn/@qhe/TCDO/runs/52aik6kr/chart)。

双卡测试对较大的模型启用冻结基座的 CPU/GPU 交接，仅 LoRA 和优化器常驻显存。DeepSeek 的测试 Actor 显存预算从 0.75 调至 0.65 后通过。Gemma 首次初始化遇到 TCP 端口争用 `EADDRINUSE`，清理该测试遗留的 EngineCore 后重新完成两步并通过全部检查。模型基础权重全部保留，测试检查点及临时适配器已清理。

GPU 环境为 PyTorch 2.10.0+cu129、Transformers 5.9.0、PEFT 0.21.0、vLLM 0.19.1、DeepSpeed 0.19.7。Qwen3.5 Learner 缺少 Transformers 线性注意力 fast-path 依赖，使用原生 PyTorch 实现完成训练；Actor 仍使用 vLLM CUDA Graph。GPU 汇总保存在 `tests/outputs/tca_four_models_gpu_20261125/gpu_validation_summary.json`，包含逐层 LoRA 检查、梯度及显存采样证据。远程执行的 60 份生产源码及配置与 Windows 工作区逐一核对一致，结果和日志已同步至 Windows 与 WSL。

完整架构在 meta device 上检查层名、LoRA r=64/alpha=128 的 A/B 矩阵维度及冻结状态。各原生架构的缩小版本在 CPU 检查 TCA top-k=512 的前向、反向、有限梯度、LoRA 更新，以及基础参数保持冻结；同时核对手工输出头与原生 logits 一致，包含 Gemma 的 softcap。官方聊天模板用于检查 Student 与特权 Teacher 的提示构造。

Qwen3.5 使用 multimodal loader，LoRA 仅作用于文本解码器。除普通注意力和 MLP，还包含线性注意力的 `in_proj_qkv`、`in_proj_z`、`in_proj_b`、`in_proj_a`、`out_proj`，覆盖 32 层的 248 个模块。保留其原生 vLLM 架构及混合注意力权重映射。[官方模型配置](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json)

Qwen3、Qwen3.5 和 Gemma 配置关闭 thinking；DeepSeek 使用原生推理模板，其模板没有 `enable_thinking` 开关。

vLLM 检查使用官方 `v0.19.1` 源码的架构注册、LoRA 支持声明和前缀/融合投影映射，未导入或启动 GPU 引擎。[vLLM Qwen3.5 源码](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/model_executor/models/qwen3_5.py)

测试环境为 PyTorch 2.10.0+cpu、Transformers 5.9.0、PEFT 0.21.0。元数据按官方 revision 和文件 hash 固定。测试源码与结果仅保留在 `tests/`，不提交 GitHub；汇总为 `tests/outputs/tca_four_models_20261125/validation_summary.json`。
