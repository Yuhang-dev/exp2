# FlashPrefill：128K 位置误差与末尾保护实验

同一模型 BF16、batch=1。主实验全长度统一 YaRN 4×；32K 另做原始 RoPE 对照。各输入由不同 gov_report 文档拼接，流内不重复文档；相同样本在各长度取同一 token 流的前缀。

attention 处理完整输入序列。MLP/RMSNorm 按 token 分块；不保留跨层 KV cache。下表是完整模型前向加末位置 LM head/argmax 的耗时，不是含持久 KV 的生成 TTFT，也不测 decode。

kernel 数值检查见 kernel_check.json，逐 token 算子分块检查和完整配置见各目录 metadata.json。

## 速度与密度

| RoPE | tokens | 方法 | 前向 ms | 加速比 | 峰值 GiB | 保留块 | attention profile ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| native | 32768 | dense | 4580.8 | 1.00× | 15.80 | 100.0% | 1418.0 |
| native | 32768 | flashprefill_tail0 | 3509.7 | 1.31× | 15.80 | 15.6% | 344.1 |
| native | 32768 | flashprefill_tail1 | 3575.7 | 1.28× | 15.80 | 16.3% | 411.1 |
| native | 32768 | flashprefill_tail2 | 3587.7 | 1.28× | 15.80 | 17.0% | 422.8 |
| yarn4 | 4096 | dense | 413.7 | 1.00× | 14.44 | 100.0% | 25.8 |
| yarn4 | 4096 | flashprefill_tail0 | 415.5 | 1.00× | 14.56 | 67.7% | 27.3 |
| yarn4 | 4096 | flashprefill_tail1 | 417.2 | 0.99× | 14.56 | 70.1% | 30.0 |
| yarn4 | 4096 | flashprefill_tail2 | 417.5 | 0.99× | 14.56 | 72.7% | 31.1 |
| yarn4 | 8192 | dense | 878.6 | 1.00× | 14.60 | 100.0% | 96.0 |
| yarn4 | 8192 | flashprefill_tail0 | 852.1 | 1.03× | 14.68 | 47.5% | 70.1 |
| yarn4 | 8192 | flashprefill_tail1 | 859.3 | 1.02× | 14.68 | 49.7% | 79.8 |
| yarn4 | 8192 | flashprefill_tail2 | 860.9 | 1.02× | 14.68 | 51.9% | 81.8 |
| yarn4 | 16384 | dense | 1945.5 | 1.00× | 15.00 | 100.0% | 375.7 |
| yarn4 | 16384 | flashprefill_tail0 | 1749.0 | 1.11× | 15.00 | 31.6% | 176.9 |
| yarn4 | 16384 | flashprefill_tail1 | 1773.5 | 1.10× | 15.00 | 32.9% | 203.8 |
| yarn4 | 16384 | flashprefill_tail2 | 1778.8 | 1.09× | 15.00 | 34.1% | 208.5 |
| yarn4 | 32768 | dense | 4585.5 | 1.00× | 15.80 | 100.0% | 1421.1 |
| yarn4 | 32768 | flashprefill_tail0 | 3597.0 | 1.27× | 15.80 | 19.6% | 424.3 |
| yarn4 | 32768 | flashprefill_tail1 | 3655.0 | 1.25× | 15.80 | 20.3% | 487.6 |
| yarn4 | 32768 | flashprefill_tail2 | 3668.8 | 1.25× | 15.80 | 21.0% | 498.3 |
| yarn4 | 65536 | dense | 11876.9 | 1.00× | 17.41 | 100.0% | 5518.2 |
| yarn4 | 65536 | flashprefill_tail0 | 7306.1 | 1.63× | 17.41 | 11.5% | 976.3 |
| yarn4 | 65536 | flashprefill_tail1 | 7430.5 | 1.60× | 17.41 | 11.9% | 1112.1 |
| yarn4 | 65536 | flashprefill_tail2 | 7452.2 | 1.59× | 17.41 | 12.3% | 1134.5 |
| yarn4 | 131072 | dense | 34724.2 | 1.00× | 20.63 | 100.0% | 21928.4 |
| yarn4 | 131072 | flashprefill_tail0 | 14944.8 | 2.32× | 20.63 | 6.6% | 2293.0 |
| yarn4 | 131072 | flashprefill_tail1 | 15204.7 | 2.28× | 20.63 | 6.8% | 2578.3 |
| yarn4 | 131072 | flashprefill_tail2 | 15255.1 | 2.28× | 20.63 | 7.0% | 2629.9 |

计时取重复测量中位数；profile 是额外前向，含事件及块计数开销，只用于定位，未混入主计时。

## 整篇前向的逐 token 误差诊断

块路由会读取同块后续 query。以下中间位置指标用于 Dense/稀疏表示差异诊断，不作为严格自回归 PPL。正 ΔNLL 表示目标 token 的概率下降。早/中/晚各抽取 128 tokens，文档流尾部抽取 512 tokens；排除报告分隔符。

| RoPE | tokens | 方法 | 平均 ΔNLL | P95 | P99 | 最大值 | ΔNLL>1 比例 | 平均 KL | top1 一致率 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native | 32768 | flashprefill_tail0 | +0.0126 | 0.3404 | 0.8007 | 5.0524 | 0.6% | 0.02223 | 93.3% |
| native | 32768 | flashprefill_tail1 | +0.0120 | 0.3276 | 0.7477 | 5.0524 | 0.5% | 0.02009 | 93.6% |
| native | 32768 | flashprefill_tail2 | +0.0098 | 0.3099 | 0.7260 | 5.0524 | 0.4% | 0.01749 | 94.0% |
| yarn4 | 4096 | flashprefill_tail0 | +0.0084 | 0.2686 | 0.6953 | 2.4779 | 0.5% | 0.01818 | 94.2% |
| yarn4 | 4096 | flashprefill_tail1 | +0.0038 | 0.2277 | 0.6628 | 2.1214 | 0.3% | 0.01401 | 94.7% |
| yarn4 | 4096 | flashprefill_tail2 | +0.0034 | 0.2152 | 0.6234 | 2.1214 | 0.3% | 0.01218 | 95.2% |
| yarn4 | 8192 | flashprefill_tail0 | +0.0078 | 0.3387 | 0.8733 | 2.8107 | 0.7% | 0.02646 | 93.7% |
| yarn4 | 8192 | flashprefill_tail1 | +0.0087 | 0.3154 | 0.8420 | 2.8107 | 0.7% | 0.02414 | 94.0% |
| yarn4 | 8192 | flashprefill_tail2 | +0.0113 | 0.2926 | 0.8302 | 2.8107 | 0.7% | 0.02074 | 94.5% |
| yarn4 | 16384 | flashprefill_tail0 | +0.0045 | 0.3695 | 0.9139 | 7.5663 | 0.8% | 0.03858 | 92.6% |
| yarn4 | 16384 | flashprefill_tail1 | +0.0084 | 0.3594 | 0.9139 | 7.5663 | 0.8% | 0.03594 | 92.9% |
| yarn4 | 16384 | flashprefill_tail2 | +0.0116 | 0.3452 | 0.9139 | 7.5663 | 0.8% | 0.03506 | 93.2% |
| yarn4 | 32768 | flashprefill_tail0 | +0.0113 | 0.5105 | 1.2075 | 3.0209 | 1.6% | 0.04361 | 90.7% |
| yarn4 | 32768 | flashprefill_tail1 | +0.0083 | 0.4821 | 1.0897 | 3.0209 | 1.3% | 0.04010 | 91.3% |
| yarn4 | 32768 | flashprefill_tail2 | +0.0045 | 0.4386 | 1.0289 | 3.0209 | 1.1% | 0.03434 | 92.0% |
| yarn4 | 65536 | flashprefill_tail0 | +0.0131 | 0.6108 | 1.4852 | 6.3976 | 2.1% | 0.07693 | 88.7% |
| yarn4 | 65536 | flashprefill_tail1 | +0.0149 | 0.5809 | 1.4075 | 6.3976 | 2.0% | 0.07009 | 89.6% |
| yarn4 | 65536 | flashprefill_tail2 | +0.0226 | 0.5618 | 1.4075 | 6.3976 | 1.8% | 0.06628 | 89.7% |
| yarn4 | 131072 | flashprefill_tail0 | -0.0077 | 0.5927 | 1.3317 | 4.1123 | 2.0% | 0.09598 | 88.2% |
| yarn4 | 131072 | flashprefill_tail1 | -0.0081 | 0.5761 | 1.2388 | 4.1123 | 1.8% | 0.08960 | 88.4% |
| yarn4 | 131072 | flashprefill_tail2 | +0.0009 | 0.5711 | 1.2344 | 4.1123 | 1.7% | 0.08389 | 88.8% |

## 位置与保护区域

tail_band 根据预测所用 query 的位置划分，跨消融保持相同分组：last_block、penultimate_block、earlier。末尾全注意力位置仍可受历史表示变化影响。

| RoPE | tokens | 方法 | 区域 | tail band | 平均 ΔNLL | P99 |
| --- | ---: | --- | --- | --- | ---: | ---: |
| native | 32768 | flashprefill_tail0 | early | earlier | -0.0063 | 0.5315 |
| native | 32768 | flashprefill_tail0 | late | earlier | +0.0036 | 0.6550 |
| native | 32768 | flashprefill_tail0 | middle | earlier | +0.0078 | 1.0230 |
| native | 32768 | flashprefill_tail0 | tail | earlier | +0.0198 | 0.7712 |
| native | 32768 | flashprefill_tail0 | tail | last_block | +0.0203 | 0.8347 |
| native | 32768 | flashprefill_tail0 | tail | penultimate_block | +0.0229 | 1.0016 |
| native | 32768 | flashprefill_tail1 | early | earlier | -0.0063 | 0.5315 |
| native | 32768 | flashprefill_tail1 | late | earlier | +0.0036 | 0.6550 |
| native | 32768 | flashprefill_tail1 | middle | earlier | +0.0078 | 1.0230 |
| native | 32768 | flashprefill_tail1 | tail | earlier | +0.0198 | 0.7712 |
| native | 32768 | flashprefill_tail1 | tail | last_block | +0.0158 | 0.4818 |
| native | 32768 | flashprefill_tail1 | tail | penultimate_block | +0.0229 | 1.0016 |
| native | 32768 | flashprefill_tail2 | early | earlier | -0.0063 | 0.5315 |
| native | 32768 | flashprefill_tail2 | late | earlier | +0.0036 | 0.6550 |
| native | 32768 | flashprefill_tail2 | middle | earlier | +0.0078 | 1.0230 |
| native | 32768 | flashprefill_tail2 | tail | earlier | +0.0198 | 0.7712 |
| native | 32768 | flashprefill_tail2 | tail | last_block | +0.0085 | 0.4378 |
| native | 32768 | flashprefill_tail2 | tail | penultimate_block | +0.0144 | 0.7544 |
| yarn4 | 4096 | flashprefill_tail0 | early | earlier | -0.0007 | 0.1191 |
| yarn4 | 4096 | flashprefill_tail0 | middle | earlier | +0.0039 | 0.5630 |
| yarn4 | 4096 | flashprefill_tail0 | tail | earlier | +0.0072 | 0.8799 |
| yarn4 | 4096 | flashprefill_tail0 | tail | last_block | +0.0384 | 1.1303 |
| yarn4 | 4096 | flashprefill_tail0 | tail | penultimate_block | -0.0024 | 0.5911 |
| yarn4 | 4096 | flashprefill_tail1 | early | earlier | -0.0007 | 0.1191 |
| yarn4 | 4096 | flashprefill_tail1 | middle | earlier | +0.0039 | 0.5630 |
| yarn4 | 4096 | flashprefill_tail1 | tail | earlier | +0.0072 | 0.8799 |
| yarn4 | 4096 | flashprefill_tail1 | tail | last_block | +0.0078 | 0.4688 |
| yarn4 | 4096 | flashprefill_tail1 | tail | penultimate_block | -0.0024 | 0.5911 |
| yarn4 | 4096 | flashprefill_tail2 | early | earlier | -0.0007 | 0.1191 |
| yarn4 | 4096 | flashprefill_tail2 | middle | earlier | +0.0039 | 0.5630 |
| yarn4 | 4096 | flashprefill_tail2 | tail | earlier | +0.0072 | 0.8799 |
| yarn4 | 4096 | flashprefill_tail2 | tail | last_block | +0.0067 | 0.3794 |
| yarn4 | 4096 | flashprefill_tail2 | tail | penultimate_block | -0.0042 | 0.3552 |
| yarn4 | 8192 | flashprefill_tail0 | early | earlier | +0.0002 | 0.1694 |
| yarn4 | 8192 | flashprefill_tail0 | late | earlier | +0.0183 | 1.0396 |
| yarn4 | 8192 | flashprefill_tail0 | middle | earlier | +0.0011 | 0.9857 |
| yarn4 | 8192 | flashprefill_tail0 | tail | earlier | +0.0213 | 0.9110 |
| yarn4 | 8192 | flashprefill_tail0 | tail | last_block | +0.0004 | 0.7392 |
| yarn4 | 8192 | flashprefill_tail0 | tail | penultimate_block | -0.0102 | 0.6588 |
| yarn4 | 8192 | flashprefill_tail1 | early | earlier | +0.0002 | 0.1694 |
| yarn4 | 8192 | flashprefill_tail1 | late | earlier | +0.0183 | 1.0396 |
| yarn4 | 8192 | flashprefill_tail1 | middle | earlier | +0.0011 | 0.9857 |
| yarn4 | 8192 | flashprefill_tail1 | tail | earlier | +0.0213 | 0.9110 |
| yarn4 | 8192 | flashprefill_tail1 | tail | last_block | +0.0080 | 0.4840 |
| yarn4 | 8192 | flashprefill_tail1 | tail | penultimate_block | -0.0102 | 0.6588 |
| yarn4 | 8192 | flashprefill_tail2 | early | earlier | +0.0002 | 0.1694 |
| yarn4 | 8192 | flashprefill_tail2 | late | earlier | +0.0183 | 1.0396 |
| yarn4 | 8192 | flashprefill_tail2 | middle | earlier | +0.0011 | 0.9857 |
| yarn4 | 8192 | flashprefill_tail2 | tail | earlier | +0.0213 | 0.9110 |
| yarn4 | 8192 | flashprefill_tail2 | tail | last_block | -0.0015 | 0.2934 |
| yarn4 | 8192 | flashprefill_tail2 | tail | penultimate_block | +0.0158 | 0.4861 |
| yarn4 | 16384 | flashprefill_tail0 | early | earlier | +0.0071 | 0.4380 |
| yarn4 | 16384 | flashprefill_tail0 | late | earlier | +0.0127 | 1.1742 |
| yarn4 | 16384 | flashprefill_tail0 | middle | earlier | +0.0122 | 1.0536 |
| yarn4 | 16384 | flashprefill_tail0 | tail | earlier | +0.0084 | 0.8059 |
| yarn4 | 16384 | flashprefill_tail0 | tail | last_block | -0.0174 | 1.0011 |
| yarn4 | 16384 | flashprefill_tail0 | tail | penultimate_block | -0.0023 | 0.5694 |
| yarn4 | 16384 | flashprefill_tail1 | early | earlier | +0.0071 | 0.4380 |
| yarn4 | 16384 | flashprefill_tail1 | late | earlier | +0.0127 | 1.1742 |
| yarn4 | 16384 | flashprefill_tail1 | middle | earlier | +0.0122 | 1.0536 |
| yarn4 | 16384 | flashprefill_tail1 | tail | earlier | +0.0084 | 0.8059 |
| yarn4 | 16384 | flashprefill_tail1 | tail | last_block | +0.0126 | 1.2556 |
| yarn4 | 16384 | flashprefill_tail1 | tail | penultimate_block | -0.0023 | 0.5694 |
| yarn4 | 16384 | flashprefill_tail2 | early | earlier | +0.0071 | 0.4380 |
| yarn4 | 16384 | flashprefill_tail2 | late | earlier | +0.0127 | 1.1742 |
| yarn4 | 16384 | flashprefill_tail2 | middle | earlier | +0.0122 | 1.0536 |
| yarn4 | 16384 | flashprefill_tail2 | tail | earlier | +0.0084 | 0.8059 |
| yarn4 | 16384 | flashprefill_tail2 | tail | last_block | +0.0111 | 0.8259 |
| yarn4 | 16384 | flashprefill_tail2 | tail | penultimate_block | +0.0214 | 0.6799 |
| yarn4 | 32768 | flashprefill_tail0 | early | earlier | -0.0145 | 0.5713 |
| yarn4 | 32768 | flashprefill_tail0 | late | earlier | +0.0218 | 1.0496 |
| yarn4 | 32768 | flashprefill_tail0 | middle | earlier | -0.0077 | 1.2386 |
| yarn4 | 32768 | flashprefill_tail0 | tail | earlier | +0.0189 | 1.1732 |
| yarn4 | 32768 | flashprefill_tail0 | tail | last_block | +0.0075 | 1.4725 |
| yarn4 | 32768 | flashprefill_tail0 | tail | penultimate_block | +0.0329 | 1.3759 |
| yarn4 | 32768 | flashprefill_tail1 | early | earlier | -0.0145 | 0.5713 |
| yarn4 | 32768 | flashprefill_tail1 | late | earlier | +0.0218 | 1.0496 |
| yarn4 | 32768 | flashprefill_tail1 | middle | earlier | -0.0077 | 1.2386 |
| yarn4 | 32768 | flashprefill_tail1 | tail | earlier | +0.0189 | 1.1732 |
| yarn4 | 32768 | flashprefill_tail1 | tail | last_block | -0.0158 | 0.6128 |
| yarn4 | 32768 | flashprefill_tail1 | tail | penultimate_block | +0.0329 | 1.3759 |
| yarn4 | 32768 | flashprefill_tail2 | early | earlier | -0.0145 | 0.5713 |
| yarn4 | 32768 | flashprefill_tail2 | late | earlier | +0.0218 | 1.0496 |
| yarn4 | 32768 | flashprefill_tail2 | middle | earlier | -0.0077 | 1.2386 |
| yarn4 | 32768 | flashprefill_tail2 | tail | earlier | +0.0189 | 1.1732 |
| yarn4 | 32768 | flashprefill_tail2 | tail | last_block | -0.0159 | 0.5202 |
| yarn4 | 32768 | flashprefill_tail2 | tail | penultimate_block | +0.0065 | 0.8034 |
| yarn4 | 65536 | flashprefill_tail0 | early | earlier | -0.0154 | 0.8027 |
| yarn4 | 65536 | flashprefill_tail0 | late | earlier | +0.0912 | 2.9379 |
| yarn4 | 65536 | flashprefill_tail0 | middle | earlier | -0.0146 | 1.2658 |
| yarn4 | 65536 | flashprefill_tail0 | tail | earlier | +0.0220 | 1.4695 |
| yarn4 | 65536 | flashprefill_tail0 | tail | last_block | +0.0050 | 1.7718 |
| yarn4 | 65536 | flashprefill_tail0 | tail | penultimate_block | -0.0204 | 1.0910 |
| yarn4 | 65536 | flashprefill_tail1 | early | earlier | -0.0154 | 0.8027 |
| yarn4 | 65536 | flashprefill_tail1 | late | earlier | +0.0912 | 2.9379 |
| yarn4 | 65536 | flashprefill_tail1 | middle | earlier | -0.0146 | 1.2658 |
| yarn4 | 65536 | flashprefill_tail1 | tail | earlier | +0.0220 | 1.4695 |
| yarn4 | 65536 | flashprefill_tail1 | tail | last_block | +0.0196 | 1.1439 |
| yarn4 | 65536 | flashprefill_tail1 | tail | penultimate_block | -0.0204 | 1.0910 |
| yarn4 | 65536 | flashprefill_tail2 | early | earlier | -0.0154 | 0.8027 |
| yarn4 | 65536 | flashprefill_tail2 | late | earlier | +0.0912 | 2.9379 |
| yarn4 | 65536 | flashprefill_tail2 | middle | earlier | -0.0146 | 1.2658 |
| yarn4 | 65536 | flashprefill_tail2 | tail | earlier | +0.0220 | 1.4695 |
| yarn4 | 65536 | flashprefill_tail2 | tail | last_block | +0.0246 | 1.2653 |
| yarn4 | 65536 | flashprefill_tail2 | tail | penultimate_block | +0.0290 | 0.9698 |
| yarn4 | 131072 | flashprefill_tail0 | early | earlier | +0.0440 | 1.6125 |
| yarn4 | 131072 | flashprefill_tail0 | late | earlier | +0.0224 | 1.4944 |
| yarn4 | 131072 | flashprefill_tail0 | middle | earlier | -0.0371 | 1.2581 |
| yarn4 | 131072 | flashprefill_tail0 | tail | earlier | -0.0260 | 1.0312 |
| yarn4 | 131072 | flashprefill_tail0 | tail | last_block | -0.0025 | 1.5781 |
| yarn4 | 131072 | flashprefill_tail0 | tail | penultimate_block | -0.0265 | 1.1371 |
| yarn4 | 131072 | flashprefill_tail1 | early | earlier | +0.0440 | 1.6125 |
| yarn4 | 131072 | flashprefill_tail1 | late | earlier | +0.0224 | 1.4944 |
| yarn4 | 131072 | flashprefill_tail1 | middle | earlier | -0.0371 | 1.2581 |
| yarn4 | 131072 | flashprefill_tail1 | tail | earlier | -0.0260 | 1.0312 |
| yarn4 | 131072 | flashprefill_tail1 | tail | last_block | -0.0057 | 1.0683 |
| yarn4 | 131072 | flashprefill_tail1 | tail | penultimate_block | -0.0265 | 1.1371 |
| yarn4 | 131072 | flashprefill_tail2 | early | earlier | +0.0440 | 1.6125 |
| yarn4 | 131072 | flashprefill_tail2 | late | earlier | +0.0224 | 1.4944 |
| yarn4 | 131072 | flashprefill_tail2 | middle | earlier | -0.0371 | 1.2581 |
| yarn4 | 131072 | flashprefill_tail2 | tail | earlier | -0.0260 | 1.0312 |
| yarn4 | 131072 | flashprefill_tail2 | tail | last_block | +0.0014 | 1.0411 |
| yarn4 | 131072 | flashprefill_tail2 | tail | penultimate_block | +0.0299 | 1.0517 |

## 只输入前缀的预测

在文档流约 25%、50%、75% 位置截断输入，只预测下一个真实 token；目标与后续文本均不进入模型。每个样本三个 probe，单独给出与 Dense 的差异；这是小样本预测检查，不是全篇困惑度。

| RoPE | tokens | 方法 | probes | Dense NLL | NLL | ΔNLL | KL | top1 一致率 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native | 32768 | dense | 12 | 2.1283 | 2.1283 | +0.0000 | 0.00000 | 100.0% |
| native | 32768 | flashprefill_tail0 | 12 | 2.1283 | 2.1270 | -0.0013 | 0.01147 | 100.0% |
| native | 32768 | flashprefill_tail1 | 12 | 2.1283 | 2.1367 | +0.0083 | 0.00544 | 100.0% |
| native | 32768 | flashprefill_tail2 | 12 | 2.1283 | 2.1163 | -0.0120 | 0.00453 | 100.0% |
| yarn4 | 4096 | dense | 12 | 0.8458 | 0.8458 | +0.0000 | 0.00000 | 100.0% |
| yarn4 | 4096 | flashprefill_tail0 | 12 | 0.8458 | 0.9649 | +0.1191 | 0.01405 | 100.0% |
| yarn4 | 4096 | flashprefill_tail1 | 12 | 0.8458 | 0.8546 | +0.0088 | 0.00191 | 100.0% |
| yarn4 | 4096 | flashprefill_tail2 | 12 | 0.8458 | 0.8520 | +0.0062 | 0.00148 | 100.0% |
| yarn4 | 8192 | dense | 12 | 1.0065 | 1.0065 | +0.0000 | 0.00000 | 100.0% |
| yarn4 | 8192 | flashprefill_tail0 | 12 | 1.0065 | 1.1052 | +0.0987 | 0.02272 | 100.0% |
| yarn4 | 8192 | flashprefill_tail1 | 12 | 1.0065 | 0.9940 | -0.0125 | 0.00365 | 100.0% |
| yarn4 | 8192 | flashprefill_tail2 | 12 | 1.0065 | 0.9821 | -0.0245 | 0.00239 | 100.0% |
| yarn4 | 16384 | dense | 12 | 2.1222 | 2.1222 | +0.0000 | 0.00000 | 100.0% |
| yarn4 | 16384 | flashprefill_tail0 | 12 | 2.1222 | 2.3161 | +0.1939 | 0.08995 | 83.3% |
| yarn4 | 16384 | flashprefill_tail1 | 12 | 2.1222 | 2.2057 | +0.0835 | 0.01490 | 91.7% |
| yarn4 | 16384 | flashprefill_tail2 | 12 | 2.1222 | 2.1665 | +0.0442 | 0.01176 | 100.0% |
| yarn4 | 32768 | dense | 12 | 2.0738 | 2.0738 | +0.0000 | 0.00000 | 100.0% |
| yarn4 | 32768 | flashprefill_tail0 | 12 | 2.0738 | 2.1078 | +0.0340 | 0.03463 | 100.0% |
| yarn4 | 32768 | flashprefill_tail1 | 12 | 2.0738 | 2.0748 | +0.0010 | 0.01956 | 83.3% |
| yarn4 | 32768 | flashprefill_tail2 | 12 | 2.0738 | 2.1025 | +0.0287 | 0.01340 | 100.0% |
| yarn4 | 65536 | dense | 12 | 2.2108 | 2.2108 | +0.0000 | 0.00000 | 100.0% |
| yarn4 | 65536 | flashprefill_tail0 | 12 | 2.2108 | 2.2733 | +0.0626 | 0.03522 | 91.7% |
| yarn4 | 65536 | flashprefill_tail1 | 12 | 2.2108 | 2.1955 | -0.0152 | 0.02612 | 91.7% |
| yarn4 | 65536 | flashprefill_tail2 | 12 | 2.2108 | 2.2111 | +0.0004 | 0.01786 | 91.7% |
| yarn4 | 131072 | dense | 12 | 2.7056 | 2.7056 | +0.0000 | 0.00000 | 100.0% |
| yarn4 | 131072 | flashprefill_tail0 | 12 | 2.7056 | 2.7219 | +0.0163 | 0.05021 | 83.3% |
| yarn4 | 131072 | flashprefill_tail1 | 12 | 2.7056 | 2.5708 | -0.1348 | 0.04444 | 83.3% |
| yarn4 | 131072 | flashprefill_tail2 | 12 | 2.7056 | 2.5449 | -0.1607 | 0.05488 | 75.0% |

明细：position_summary.csv、per_sample_errors.csv、worst_tokens.csv、prefix_summary.csv。P95/P99 为采样 token 的分位数，不是置信区间；重复计时不增加独立文档样本数。

归档说明：本文件保存用户于 2026-09-15 提供的远端汇总表；原报告、PNG、CSV 和检查 metadata 的远端目录为 `results/study/`。
