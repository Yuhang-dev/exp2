# FlashPrefill 来源与适配

- 上游：[qhfan/FlashPrefill](https://github.com/qhfan/FlashPrefill)
- 固定 commit：`baa612047433a992a00d07dc178205eed065ae14`
- 原文件：`ops/flashprefill_native_forward.py`
- 本地文件：`upstream/flashprefill_native_forward.py`
- Qwen2 默认参数依据：同一 commit 的 `ruler/models/qwen2.py`。

保留上游 mean pooling、V1 块评分公式、归一化、阈值选择、索引排序和 sparse attention。块评分 kernel 的改写见下文，其他计算 kernel 及 autotune 配置保持不变。

初版适配删除 `fla.utils` 的导入及 `@contiguous`、`@autocast_custom_fwd` 两个装饰器。`attention.py` 显式提供连续的 Q/K/V 和输出张量，模型固定 BF16 推理，因此不需要安装 FLA。

本次核对先修正 `compute_block_score` 的 query 有效性掩码，排除不完整 query block 的补齐行；这一项不改变整块长度的评分公式。

随后远端 1K 数值检查发现原始块评分输出出现重复列和未来块非零值，eager/compiled 归一化一致，问题在归一化前已发生。现将 `Q @ K_mean.T` 后的 `[Q,K,1]` reshape/列归约，改写成数学等价的 `K_mean @ Q.T` 加行归约，仍用 `tl.dot` GEMM、原来的 mean-k、因果可见性及评分公式。该改写针对本项目 `K_STRIDE=BLOCK_SIZE=128`，实现标识为 `v1_kmean_qt_row_reduce`。未据此断言具体是 Triton 哪一个 compiler pass 的缺陷。

attention 的 QK/softmax/PV kernel 和 autotune 候选参数没有修改。旧 4K–32K 数字暂不作为通过正确性验证的 FlashPrefill 基线；`run_study.sh` 在数值检查通过后会先重跑旧生成基线，再运行位置误差与 128K 实验。检查范围及证据见 [KERNEL_AUDIT.md](KERNEL_AUDIT.md)。

采用 V1 的纯稀疏路径，没有 V2 mean correction。块大小 128，alpha=0.08，sink=2 blocks，window=4 blocks，最后 2 个 query blocks 保留完整历史，min_budget=0。全部 28 层使用上述 prefill 配置。

`run_study.sh` 额外对比末尾全注意力块为 0、1、2 的设置；新实验的主曲线统一 YaRN 4×，并另做相同 32K 输入的原始 RoPE 对照。配置不改变模型权重。

exp1 的 Transformers 为 4.51.3，因此接入该版本的 attention dispatch，继续使用库内原始 Qwen2 投影、RoPE、KV cache 和输出投影。两条路径的 decode 都使用 PyTorch Flash SDPA，enable_gqa=True；prefill 使用因果模式，单 token decode 可以读取全部已有 KV。
