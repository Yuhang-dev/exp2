# FlashPrefill 来源与适配

- 上游：[qhfan/FlashPrefill](https://github.com/qhfan/FlashPrefill)
- 固定 commit：`baa612047433a992a00d07dc178205eed065ae14`
- 原文件：`ops/flashprefill_native_forward.py`
- 本地文件：`upstream/flashprefill_native_forward.py`
- Qwen2 默认参数依据：同一 commit 的 `ruler/models/qwen2.py`。

保留上游 mean pooling、块评分、归一化、阈值选择、索引排序和 sparse attention kernels，包含原有 autotune 和 torch.compile。

适配只删除 `fla.utils` 的导入及 `@contiguous`、`@autocast_custom_fwd` 两个装饰器。`attention.py` 显式提供连续的 Q/K/V 和输出张量，模型固定 BF16 推理，因此不需要安装 FLA。

采用 V1 的纯稀疏路径，没有 V2 mean correction。块大小 128，alpha=0.08，sink=2 blocks，window=4 blocks，最后 2 个 query blocks 保留完整历史，min_budget=0。全部 28 层使用上述 prefill 配置。

exp1 的 Transformers 为 4.51.3，因此接入该版本的 attention dispatch，继续使用库内原始 Qwen2 投影、RoPE、KV cache 和输出投影。两条路径的 decode 都使用 PyTorch Flash SDPA，enable_gqa=True；prefill 使用因果模式，单 token decode 可以读取全部已有 KV。
