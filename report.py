"""Summarize paired measurements and draw speed/memory/quality figures."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_report(folder):
    timings = pd.read_csv(folder / "timings.csv")
    quality = pd.read_csv(folder / "quality.csv")
    layers = pd.read_csv(folder / "layers.csv")
    summary = timings.groupby(["seq_len", "method"]).agg(
        measurements=("prefill_ms", "size"),
        prefill_ms=("prefill_ms", "median"),
        prefill_min_ms=("prefill_ms", "min"),
        prefill_max_ms=("prefill_ms", "max"),
        decode_tokens_s=("decode_tokens_s", "median"),
        generation_peak_gib=("generation_peak_gib", "max"),
        prefill_peak_gib=("prefill_peak_gib", "max"),
    )
    qsummary = quality.groupby(["seq_len", "method"]).agg(
        tail_nll=("tail_nll", "mean"),
        next_token_kl=("next_token_kl_dense_to_method", "mean"),
        next_token_agreement=("same_next_token_as_dense", "mean"),
    )
    qsummary["tail_ppl"] = np.exp(qsummary["tail_nll"])
    density = layers.groupby(["seq_len", "method"])["causal_block_density"].mean()
    attention = layers.groupby(["seq_len", "method", "sample"])["attention_ms"].sum()
    attention = attention.groupby(["seq_len", "method"]).median().rename("profile_attention_ms")
    summary = summary.join(qsummary).join(density).join(attention).reset_index()
    summary.to_csv(folder / "summary.csv", index=False)

    dense = summary[summary.method == "dense"].set_index("seq_len")
    sparse = summary[summary.method == "flashprefill"].set_index("seq_len")
    comparison = pd.DataFrame({
        "dense_prefill_ms": dense.prefill_ms,
        "flashprefill_ms": sparse.prefill_ms,
        "prefill_speedup": dense.prefill_ms / sparse.prefill_ms,
        "dense_decode_tokens_s": dense.decode_tokens_s,
        "flashprefill_decode_tokens_s": sparse.decode_tokens_s,
        "dense_peak_gib": dense.generation_peak_gib,
        "flashprefill_peak_gib": sparse.generation_peak_gib,
        "dense_tail_nll": dense.tail_nll,
        "flashprefill_tail_nll": sparse.tail_nll,
        "delta_tail_nll": sparse.tail_nll - dense.tail_nll,
        "next_token_kl": sparse.next_token_kl,
        "flashprefill_block_density": sparse.causal_block_density,
    })
    comparison.to_csv(folder / "comparison.csv")

    plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 10})
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    specs = [("prefill_ms", "Prefill / TTFT (ms)"),
             ("decode_tokens_s", "Decode (tokens/s)"),
             ("generation_peak_gib", "Peak allocated memory (GiB)")]
    for ax, (column, title) in zip(axes, specs):
        for method, label, color in [("dense", "Dense Flash SDPA", "#3569a8"),
                                      ("flashprefill", "FlashPrefill V1", "#e27739")]:
            data = summary[summary.method == method]
            ax.plot(data.seq_len / 1024, data[column], "o-", label=label, color=color)
        ax.set(title=title, xlabel="Prompt length (Ki tokens)")
        ax.set_xticks(dense.index.to_numpy() / 1024)
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    figure.savefig(folder / "performance.png")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    x = comparison.index.to_numpy() / 1024
    axes[0].plot(x, dense.tail_nll, "o-", label="Dense Flash SDPA", color="#3569a8")
    axes[0].plot(x, sparse.tail_nll, "o-", label="FlashPrefill V1", color="#e27739")
    axes[0].set_title("Document-tail NLL")
    axes[0].legend(frameon=False)
    axes[1].plot(x, comparison.next_token_kl, "o-", color="#8064a2")
    axes[1].set_title("Next-token KL (dense || sparse)")
    axes[2].plot(x, comparison.flashprefill_block_density * 100, "o-", color="#27856d")
    axes[2].set_title("Selected causal blocks (%)")
    for ax in axes:
        ax.set_xlabel("Prompt length (Ki tokens)")
        ax.set_xticks(x)
        ax.grid(alpha=0.2)
    figure.savefig(folder / "quality.png")
    plt.close(figure)

    lines = [
        "# Qwen2.5-7B：FlashPrefill V1 对照结果", "",
        "同一批 LongBench gov_report 文档、BF16、batch=1。每种长度使用相同文档的前缀，"
        "两种方法共用同一组输入 token。速度使用重复测量的中位数；峰值显存取最大值。", "",
        "| 输入 tokens | Dense prefill ms | FlashPrefill ms | 加速比 | Dense decode tok/s | FlashPrefill decode tok/s |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for length, row in comparison.iterrows():
        lines.append(f"| {length} | {row.dense_prefill_ms:.1f} | {row.flashprefill_ms:.1f} | "
                     f"{row.prefill_speedup:.2f}× | {row.dense_decode_tokens_s:.1f} | "
                     f"{row.flashprefill_decode_tokens_s:.1f} |")
    lines += ["", "| 输入 tokens | Dense peak GiB | FlashPrefill peak GiB | Dense tail NLL | FlashPrefill tail NLL | ΔNLL | 保留块比例 |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for length, row in comparison.iterrows():
        lines.append(f"| {length} | {row.dense_peak_gib:.2f} | {row.flashprefill_peak_gib:.2f} | "
                     f"{row.dense_tail_nll:.4f} | {row.flashprefill_tail_nll:.4f} | "
                     f"{row.delta_tail_nll:+.4f} | {row.flashprefill_block_density:.1%} |")
    lines += ["", "![速度与显存](performance.png)", "", "![近似影响](quality.png)", "",
              "计时从 GPU 上已有的输入开始，包含完整模型 prefill、最后一个位置的 LM head 和贪心选词；"
              "不含模型加载、tokenization、输入拷贝、首次编译和 autotune。", "",
              "Prefill 产生第一个 token，随后 N−1 次 dense decode 产生剩余 token。为固定测量工作量，"
              "不在 EOS 处提前停止。Decode 计时包含缓存更新和 Python 调度。", "",
              "tail NLL 只评价截取文档最后一段 token，不包含指令和 chat template；它不是全篇困惑度或任务准确率。"
              "原始质量测量及 token 数见 quality.csv。", "",
              "layers.csv 来自单独的质量/profile 前向，含事件和块计数记录开销，不纳入主计时。"
              "块比例以因果可见的 128×128 块为分母，包含强制保留的 sink、局部窗口和末尾 query 块。", "",
              "复现信息见 metadata.json；输入文档 ID、长度和 token SHA256 见 inputs.jsonl。", ""]
    (folder / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    make_report(parser.parse_args().folder)
