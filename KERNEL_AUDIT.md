# Kernel 核对与数值检查

上游固定为 `qhfan/FlashPrefill@baa612047433a992a00d07dc178205eed065ae14`。环境为 PyTorch 2.6.0+cu124、Triton 3.2.0、Transformers 4.51.3；模型固定 Qwen2.5-7B-Instruct 的 28 个 Q heads、4 个 KV heads、head_dim=128、BF16。

## 源码核对

源码核对之后，远端数值检查发现了下述块评分错误；仅看源码不能判定 kernel 已正确执行。当前改写的 GPU 结果等待重新验证。

| 项目 | 核对结果 |
| --- | --- |
| 张量布局 | 传入上游的是 `[B,N,H,D]`，stride 按该布局传递；上游最外层函数的 `(b h s d)` 注释不准确，调用实现按 `(b s h d)` 使用 |
| GQA | Q head `h` 对应 KV head `h//7`；mean-k、评分及稀疏 attention 一致 |
| 缩放 | QK 乘 `1/sqrt(128)`；exp2 前额外乘 `log2(e)`，与 softmax 定义一致 |
| 因果计算 | 选择器排除未来 key blocks；对角 block 在 attention 内逐 token 施加 `key_position <= query_position` |
| 归一化 | 在线 softmax 保持 FP32 的最大值、分母和输出累加；P 转 BF16 后进行 PV，需按 BF16 容差检查 |
| 首次累加 | `m=-inf,l=1` 在首个有可见 key 的 tile 得到缩放系数 0，旧分母被清零；sink/局部窗口保证有效 query 有可见 key |
| 路由 | mean-k 评分公式、相对最大分数阈值、sink/窗口/末尾保护、因果交集及排序与上游一致；评分 kernel 的原始数值输出曾验证失败，已改写 |
| Decode | 原基线单 token decode 使用全历史 SDPA，`is_causal=False`；不使用左上对齐三角掩码 |

发现并修正一个边界问题：当 N 不是 128 的倍数时，`compute_block_score` 会把补齐的零 query 也计入历史块分数。现在其 `causal_mask` 同时要求 `q_index < query_len`。原先 4096/8192/16384/32768 的实验长度均整块，此修正不改变它们的公式。

本地 NumPy 数学反例已验证：65 个有效 query 对两个历史块的 logits 为 `[2,-2]`，原逻辑额外计入 63 个补齐 query 后，归一化分数由 `[0.9820,0.0180]` 变成 `[0.8833,0.1167]`；alpha=0.08 时第二块由不选变为选中。这验证了边界问题能改变路由；GPU kernel 的数值验证另由下述脚本执行。

## 远端发现的评分数值错误

用户在 RTX 4090 / Triton 3.2.0 上运行 `debug_scores.py` 的输出保存在 [score_failure.jsonl](results/kernel_diagnostics/score_failure.jsonl)。N=1024、Q=0 时，raw score 的数学期望为：query block 之前的每个完整 key block 计数 128，对角块计数 1，未来块计数 0。

实际第 1 个 query block（从 0 编号）的 raw score 为 `[64,0,0,0,64,0,0,0]`，期望为 `[128,1,0,0,0,0,0,0]`。head 0 的未来区域存在 12 个有限 max 值，期望为 0 个。eager 与 compiled 归一化在零 Q 下完全相同，随机 Q 下最大差约 2.98e-8；所以偏差已发生于归一化前的 `compute_block_score`。

当前改写将矩阵乘法改为 `K_mean @ Q.T`，每行对应一个 key block，直接沿 query 维做 max/sum。它等价于原 V1 公式，仍使用 GEMM；去掉了原 `[Q,K,1]` reshape 和 axis-0 归约路径。具体底层 compiler pass 的责任未定位，不以此推断论文算法本身存在同样错误。

本地 NumPy 对照覆盖 N=1024/1089 的零 Q 与随机 Q，比较原公式的全局归一化和改写的逐 key-block max/sum 后归一化：零 Q 完全一致，随机输入最大分数差分别为 1.79e-7、8.94e-8。此项验证数学等价性，不替代远端 GPU 检查。

零 Q 解析计数现在对全部 20 个评分 autotune 候选逐个验证，要求精确一致；原来的随机输入 FP32 对照容差保持不变。131072 长度另外核对采样 query blocks 的 proxy score，而不仅检查 attention 索引。

上述错误在 1K 测试上已复现，旧 4K–32K 测量尚未独立证明不受影响。旧数字暂不作为已验证基线，修复通过数值检查后重跑。

## 需要区分的两种因果性

固定所选 blocks 后，attention 的读 KV 操作是逐 token 因果的。但选择哪些 blocks，是汇总一个 query block 内的 query 后决定的。同块后续 query 可以改变较早位置的路由。这是原版块选择算法的性质，不等于 attention 的三角掩码出错。

因此，整段输入前向得到的中间位置 NLL/KL 用于表示误差诊断，不能直接解释成严格自回归 PPL。此前报告中的 `tail PPL` 应同样按诊断指标理解。新实验额外在 25%/50%/75% 附近截断输入，只预测一个未输入的下一个 token，单列 prefix NLL/KL。

末尾保护也是显式实验变量：默认最后 2×128 个 query 位置读取全部历史。`tail_protected` 仅标记这一规则；首部等位置也可能因 sink/局部窗口或阈值选择而实际读取全部可见 blocks。

## GPU 数值检查

```bash
source ./env.sh
python -u check_kernel.py --out results/kernel_check.json
```

检查使用独立 PyTorch FP32 数学参考，包括：

1. N=1024 和 N=1089：全部 20 个评分候选的零 Q 解析计数；mean-k 和随机输入归一化块评分，对齐长度与不完整末块。
2. 非均匀分数下的 0/1/2 个末尾保护块：indices/counts 与独立布尔选择器精确一致。
3. 人工指定不同 head 的稀疏块：原 attention kernel 与相同稀疏掩码的 FP32 QK-softmax-PV 比较。
4. 固定掩码后改变未来 K/V，验证此前输出不变。
5. alpha=0 的全保留路径与独立完整因果 attention 对照，以及完整路由+attention 路径对照。
6. N=131072：稀疏 kernel 和评分 kernel 全长执行，抽查首部、32K/64K/128K 边界位置和 query blocks，检查大 stride/index 下的数值。
7. 一个确定构造记录“同块后续 query 改变路由”的结构性质，单独标为 OBSERVED。

attention 默认逐元素 atol=0.02、rtol=0.02，整体相对 L2 必须小于 0.02；平均向量和分数使用更小绝对容差。每项同时输出实际最大绝对误差和相对 L2，而不只给 PASS。indices/counts 和固定掩码的因果检查要求精确一致。

`run_study.sh` 先运行检查；任一断言失败就退出，不进入模型实验。通过后先重跑原始 4K–32K 生成基线到 `results/study/baseline_native`，再开始新实验。新实验模型载入后，另核验 token 分块的 MLP/RMSNorm 与未分块计算的数值。

本地可完成源码核对与语法检查。GPU 数值通过与否，以远端生成的 `kernel_check.json` 为准，当前不预填 PASS。

## 128K 的执行口径

- BF16 模型权重全驻 GPU；attention 始终一次处理完整 N 个位置，没有把 attention 改成独立窗口。
- MLP 和 RMSNorm 在 token 维上相互独立，按 1024 tokens 分块降低临时激活峰值。
- `use_cache=False`，每层计算完整 K/V 后释放，不保留 28 层的解码缓存。只测完整前向与误差，不测 128K decode，也不把此耗时与旧版 TTFT 混合。
- 显式设置 `use_sliding_window=False, sliding_window=None`。否则 Transformers 4.51.3 在长度达到原配置 `sliding_window=131072` 时，可能因 SDPA 的 mask 判断构造巨大 N×N mask，即使模型的滑窗开关关闭。
- 所有主曲线长度统一 YaRN 4×；`max_position_embeddings=131072`、`original_max_position_embeddings=32768`。该版本 YaRN 实现会由这两个值重新计算 factor，不能只写 factor=4 而保留 max=32768。
- 每个长度独立进程，释放上一长度模型与编译器分配；计时前预热全部四条路径。数据是标明来源的多文档拼接流，不声称 gov_report 单篇文档天然达到 128K。

参考：[Qwen 模型卡的长文配置](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct#processing-long-texts)、[Transformers 4.51.3 YaRN 实现](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/modeling_rope_utils.py#L235-L261)、[Qwen2 mask 实现](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2/modeling_qwen2.py)。
