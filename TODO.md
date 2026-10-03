# 手搓 Llama 4：完整代码 TODO

范围是 `../llama-models/models/llama4/` 的完整推理实现，以及它直接依赖的共享 `models/checkpoint.py`、`models/datatypes.py` 等接口。按依赖顺序编写；只勾选**代码和静态对照**，无需下载权重或运行模型。每项完成时标注对应参考类/函数及输入输出 shape。

## 0. 工程骨架与依赖

- [ ] 创建 `src/my_llama/` 的各模块与 `__init__.py`，按 [README.md](README.md) 的目录分层。
- [ ] 搞清楚模型各个模块的调用关系（按照完整输入输出的角度）
- [ ] 写 `pyproject.toml`：Python、PyTorch、torchvision、FairScale、tiktoken、Pydantic、Pillow、Fire、量化相关依赖；声明 completion/chat/quantize CLI 入口。
- [ ] 梳理共享类型：`RawMessage`、`RawContent`、`RawMediaItem`、`StopReason`、`GenerationResult`、`QuantizationMode`、工具调用格式；决定在本项目定义或明确引用。
- [ ] 为每个模块写参考源码路径、关键 shape 和外部依赖注释，避免漏掉隐藏的加载 hook。

## 1. 参数与数据结构

- [ ] 复现 `QuantizationArgs`、`LoRAArgs`、`MoEArgs`、`VisionArgs`、`ModelArgs` 的字段与默认值。
- [ ] 复现模型参数校验、scaled RoPE 默认参数，以及 Scout/Maverick 使用到的配置分支。
- [ ] 复现 `LLMInput`、`TransformerInput`、`MaskedEmbedding`、`LLMOutput`，写清 batch、序列、tile 和 patch 维度。

## 2. 模型并行与文本基础层

- [ ] 实现或封装 distributed/NCCL 和 FairScale tensor parallel 初始化；明确 world size、rank 与本地 GPU 的关系。
- [ ] 封装列并行、行并行线性层、词表并行 embedding 和归约操作；保留 checkpoint 对应的参数布局。
- [ ] 复现 float32 统计的 RMSNorm、SwiGLU FeedForward 及 dense FFN 隐藏维计算。
- [ ] 复现 RoPE 频率预计算、Scout scaled RoPE、复数旋转和 broadcast 规则。

## 3. Attention 与 Transformer

- [ ] 复现 Q/K/V/O 投影、GQA 头复用、可选 Q/K norm、RoPE/NoPE 选择。
- [ ] 复现 NoPE attention temperature tuning 的位置与公式。
- [ ] 复现逐层 K/V cache 的分配、写入、读取、位置切片与 batch 上限。
- [ ] 复现全局因果 mask、chunk 局部因果 mask，以及 prefill/decode 时 mask 的选择。
- [ ] 复现 `TransformerBlock` 的 pre-norm 残差次序和 dense FFN/MoE 交替规则。
- [ ] 复现 `Transformer` 的 embedding、全部 block、最终 norm、输出投影及视觉 embedding 替换入口。
- [ ] 复现原生权重 `load_state_dict` hooks：合并 QKV 拆分、层 norm 重命名和无用 extra state 处理。

## 4. MoE

- [ ] 复现 `Experts` 的三组专家权重布局、batched SwiGLU 和模型并行维度切分。
- [ ] 复现 top-k router：logit 计算、未选中项屏蔽、sigmoid 权重、token 展开和专家批处理。
- [ ] 复现共享专家与路由专家的结果合并、`scatter_add_` 和模型并行归约。
- [ ] 复现 MoE/shared FFN 的原生权重重命名、reshape hook。

## 5. Tokenizer 与对话格式

- [ ] 复现 tiktoken BPE 文件加载、正则模式、全部 Llama 4 特殊 token 排列与 ID。
- [ ] 复现 BOS/EOS、pad、`<|eom|>`、`<|eot|>`、stop tokens 和 encode/decode 行为。
- [ ] 复现 system/user/assistant/tool 消息头、消息结束符、多轮 prompt 和 assistant 开头。
- [ ] 复现文本/图片交错内容、工具调用的编码与 assistant 输出解码。
- [ ] 复现图片相关 token：`<|image_start|>`、`<|image|>`、`<|patch|>`、tile 分隔符、`<|image_end|>`；由视觉 patch 数推导占位数。

## 6. 视觉预处理与编码

- [ ] 复现图片 RGB 转换、静态 resize/normalize、动态候选分辨率、保持比例 resize、pad、tile 切分。
- [ ] 复现多 tile 图片额外的全局缩放图，以及 tile 顺序和 aspect ratio 信息。
- [ ] 复现视觉 patch embedding、class token、位置编码、2D RoPE、视觉 Transformer block 与中间层特征拼接。
- [ ] 复现 pixel shuffle、视觉 MLP、视觉到语言维度投影。
- [ ] 复现按 `<|patch|>` mask 将图像 embedding 散射回文本序列；覆盖 batch 内多图、多 tile 的索引逻辑。
- [ ] 复现视觉权重加载 hook，包括空 shape 参数的恢复逻辑。

## 7. 生成与原生 checkpoint

- [ ] 复现 `Llama4.build`：进程组/模型并行初始化、设备与 seed、`params.json`、tokenizer/词表校验。
- [ ] 复现 `.pth` 分片发现、rank 映射、原始/目标并行度重排以及 MoE 权重转换。
- [ ] 复现普通 BF16/FP16 模型实例化和 state dict 装载路径。
- [ ] 复现 prefill、增量 decode、不同长度 prompt 的 batch padding、图像只在首轮编码。
- [ ] 复现 greedy、temperature + top-p、可选 logits processor、echo、logprobs、停止 token 和流式 `GenerationResult`。
- [ ] 复现 `completion` 与 `chat_completion` 两个对外生成接口。

## 8. 量化与命令行入口

- [ ] 复现 `quantization/loader.py` 中 FP8 mixed、Int4 mixed 的模块转换、选择规则和量化权重装载。
- [ ] 复现 `scripts/quantize.py` 的量化权重生成、checkpoint 读写和命令行参数。
- [ ] 复现 `scripts/completion.py`、`scripts/chat_completion.py` 的参数、调用链与文本/图文输入示例。
- [ ] 将量化模式贯通配置、`Llama4.build`、loader 和 CLI。

## 9. 最后做一次静态覆盖审阅

- [ ] 对照参考目录逐文件检查：所有公开类/函数、关键分支、权重加载 hook、CLI 入口均有实现位置。
- [ ] 检查张量 shape、转置/reshape 方向、模型并行切分与归约位置的代码注释。
- [ ] 记录任何与参考实现不同的设计及原因；未实现项留在本清单中，不把“完整版”标记完成。
