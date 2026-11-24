# TCA 模型代码兼容性检查

2026-11-25 在 WSL 的 `py312-th210-vl019-cu128` 环境执行 20 项 CPU/静态检查，全部通过。没有下载或加载预训练权重，没有启动 vLLM 引擎、训练任务或 GPU 任务。

| 模型 | 原生架构 | 完整架构 LoRA 模块数 | 配置 |
| --- | --- | ---: | --- |
| Qwen/Qwen3-4B | Qwen3ForCausalLM | 252 | `configs/training/tca/qwen3_4b_100steps_8GPU.yaml` |
| Qwen/Qwen3.5-4B | Qwen3_5ForConditionalGeneration | 248 | `configs/training/tca/qwen3_5_4b_100steps_8GPU.yaml` |
| google/gemma-4-E4B-it | Gemma4ForConditionalGeneration | 258 | `configs/training/tca/gemma4_e4b_it_100steps_8GPU.yaml` |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | Qwen2ForCausalLM | 196 | `configs/training/tca/deepseek_r1_distill_qwen_1p5b_100steps_8GPU.yaml` |

完整架构在 meta device 上检查层名、LoRA r=64/alpha=128 的 A/B 矩阵维度及冻结状态。各原生架构的缩小版本在 CPU 检查 TCA top-k=512 的前向、反向、有限梯度、LoRA 更新，以及基础参数保持冻结；同时核对手工输出头与原生 logits 一致，包含 Gemma 的 softcap。官方聊天模板用于检查 Student 与特权 Teacher 的提示构造。

Qwen3.5 使用 multimodal loader，LoRA 仅作用于文本解码器。除普通注意力和 MLP，还包含线性注意力的 `in_proj_qkv`、`in_proj_z`、`in_proj_b`、`in_proj_a`、`out_proj`，覆盖 32 层的 248 个模块。保留其原生 vLLM 架构及混合注意力权重映射。[官方模型配置](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json)

Qwen3、Qwen3.5 和 Gemma 配置关闭 thinking；DeepSeek 使用原生推理模板，其模板没有 `enable_thinking` 开关。

vLLM 检查使用官方 `v0.19.1` 源码的架构注册、LoRA 支持声明和前缀/融合投影映射，未导入或启动 GPU 引擎。[vLLM Qwen3.5 源码](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/model_executor/models/qwen3_5.py)

测试环境为 PyTorch 2.10.0+cpu、Transformers 5.9.0、PEFT 0.21.0。元数据按官方 revision 和文件 hash 固定。测试源码与结果仅保留在 `tests/`，不提交 GitHub；汇总为 `tests/outputs/tca_four_models_20261125/validation_summary.json`。
