# Qwen2.5-7B：原版 FlashPrefill 长文基线

对比原版 **FlashPrefill V1** 与 **PyTorch Flash SDPA 全注意力**，复用 exp1 的模型、LongBench 缓存和 Conda 环境。默认模型为 `Qwen/Qwen2.5-7B-Instruct`，BF16、batch=1，单张 RTX 4090 24GB。

## 在远端运行

扩展至 128K 的 kernel 检查、位置误差和末尾保护消融：

```bash
cd /root/autodl-tmp/exp2
git pull
bash run_study.sh
```

脚本先做 GPU kernel 数值检查，通过后先重跑原始 4K–32K 生成基线，再运行 4K/8K/16K/32K/64K/128K 的位置实验，统一 YaRN 4×；每种长度 4 个文档流、每种方法 3 次计时，比较 Dense 与末尾保护 0/1/2 blocks 的 FlashPrefill。最后额外运行原始 RoPE 的 32K 对照。全部复用 exp1 环境。

如果 kernel 检查、生成基线和文档流准备已经完成，直接运行位置实验，保留已有基线：

```bash
cd /root/autodl-tmp/exp2
git pull
bash run_positions.sh
```

这个入口使用现有 `results/study/streams.pt`，从 4K 位置实验开始，随后运行到 128K。

新报告：`results/study/REPORT.md`，每完成一个长度更新一次；重跑的生成基线在 `results/study/baseline_native/REPORT.md`。逐 token 误差、位置分组、P95/P99、最差 token 上下文及严格前缀预测分别保存在对应 CSV。大文件 `streams.pt` 和逐次 `tokens.csv` 留在远端，不提交 Git。

为在 24 GiB GPU 上处理 128K，新实验把 MLP/RMSNorm 按 token 分块，并使用 `use_cache=False`；attention 仍处理完整序列。新速度指标是**不保留跨层 KV 的完整模型前向时间**，与下方旧实验的生成 TTFT 分开。核对发现的不完整 query block 评分问题、块路由的同块后续 query 依赖，以及检查项目详见 [KERNEL_AUDIT.md](KERNEL_AUDIT.md)。

原版 4K–32K 生成基线仍使用下面的入口：

```bash
cd /root/autodl-tmp
git clone https://github.com/Yuhang-dev/exp2.git
cd exp2
bash run.sh
```

现有环境：`/root/autodl-tmp/conda/envs/exp1`，PyTorch `2.6.0+cu124`、Transformers `4.51.3`、Triton `3.2.0`。运行脚本直接 source `/root/autodl-tmp/exp1/env.sh` 并激活这个环境，无需额外安装。FlashPrefill 上游来源和适配见 [UPSTREAM.md](UPSTREAM.md)。

首次运行每种长度会进行 Triton autotune 和 torch.compile；终端显示 `Warmup` 的部分不计入结果。

默认输出 `results/qwen25_7b/REPORT.md`。更改实验规模或输出目录：

```bash
bash run.sh --lengths 4096 8192 16384 32768 --samples 2 --repeats 3 \
  --new-tokens 64 --quality-tokens 512 --out results/qwen25_7b
```

仅执行一个长度：

```bash
bash run.sh --lengths 8192 --samples 1 --repeats 3 --out results/qwen25_8k
```

## 实验定义

| 项目 | 配置 |
| --- | --- |
| 输入长度 | 4,096、8,192、16,384、32,768 tokens，包含 chat template |
| 文档 | LongBench gov_report，seed=42，选 2 篇足够长的文档 |
| 配对方式 | 各长度复用相同文档；Dense 与 FlashPrefill 输入完全一致 |
| 重复测量 | 每种方法、每个文档、每个长度 3 次，交替执行顺序 |
| 生成 | 贪心固定 64 tokens；第 1 个来自 prefill，后 63 个来自 decode |
| FlashPrefill | block=128、alpha=0.08、sink=2、window=4、last_query_blocks_full=2 |
| Decode | 两条路径均使用全注意力 Flash SDPA，保留完整 KV |
| 权重与位置编码 | 同一模型权重；原始 RoPE，不做训练或长度扩展 |

计时使用 CUDA synchronize 包围的墙钟时间，覆盖完整模型，不含模型加载、tokenization、H2D 和编译。记录 prefill/TTFT、prefill tokens/s、decode tokens/s、decode ms/token、完整生成时间、峰值 allocated/reserved 显存。长序列仅计算最后位置的 LM head，避免创建 N×词表大小的 logits。

精度代理包括文档末尾 512 tokens 的 teacher-forced NLL/PPL，以及输入结束位置的 next-token KL 和 argmax 一致性。NLL 在文档区域取样，不计末尾指令；LM head 分块计算。截取文档不做摘要答案评分。

核对后明确：块路由会汇总同块后续 query，因此上述整篇前向的中间位置 NLL/PPL 应解释为误差诊断，不能作为严格自回归困惑度。新实验另外提供只输入前缀的预测对照。

另做一次独立 profile，保存每层 attention 时间及保留块比例。profile 的事件和计数开销不混入主计时。速度表取重复测量中位数，显存取最大值；默认 2 篇文档的结果用于建立基线。

## 输出

| 文件 | 内容 |
| --- | --- |
| `REPORT.md`、`performance.png`、`quality.png` | 对照表和图 |
| `timings.csv` | 每次运行的原始速度和显存 |
| `quality.csv` | 每篇文档的 NLL/PPL、next-token KL |
| `layers.csv` | 每层 attention profile、因果块保留比例 |
| `summary.csv`、`comparison.csv` | 聚合指标和加速比 |
| `generations.jsonl` | 每次生成的 token IDs 和文本 |
| `metadata.json` | GPU、环境版本、参数、模型和代码 commit |
| `inputs.jsonl`、`inputs.pt` | 文档 ID/输入 SHA256，以及本机保存的精确输入 tokens |

`inputs.pt` 不提交到 Git；可从已有缓存按同一 seed 和 metadata 重建。重新生成报告：

```bash
source ./env.sh
python report.py results/qwen25_7b
```

## 执行状态

用户远端已完成 kernel 检查、修复后的 4K–32K 生成基线、YaRN 4× 的 4K–128K 位置实验及原始 RoPE 32K 对照。默认保护最后 2 个块时，128K 完整前向由 34.72 s 降至 15.26 s（2.28×），峰值显存 20.63 GiB；采样 token 平均 ΔNLL 为 +0.0009，P99 为 1.2344。完整表格见[远端报告归档](results/study_4090/REPORT.md)，实验解读及下一步定位依据见 [RESULTS.md](RESULTS.md)。这些误差指标和检查结果的适用范围以各自定义和测试用例为准。

最差 token 和分样本 CSV 也已归档，具体失败位置与复现候选见 [FAILURE_ANALYSIS.md](FAILURE_ANALYSIS.md)。`python analyze_failures.py` 可重新生成加权样本统计和候选 JSON；此命令只分析 CSV，不执行模型。
