"""Aggregate completed position/guard studies; keep native RoPE and YaRN separate."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def report(folder):
    tables = {name: [] for name in ("timings", "tokens", "prefix", "layers")}
    for marker in sorted(folder.glob("*/completed.json")):
        run = marker.parent
        metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        for name in tables:
            frame = pd.read_csv(run / f"{name}.csv")
            frame["rope"] = metadata["arguments"]["rope"]
            tables[name].append(frame)
    timing, tokens, prefix, layers = [pd.concat(tables[name], ignore_index=True) for name in tables]
    group = ["rope", "seq_len", "method"]
    speed = timing.groupby(group).agg(forward_ms=("forward_ms", "median"),
        min_ms=("forward_ms", "min"), max_ms=("forward_ms", "max"),
        peak_gib=("peak_gib", "max"), measurements=("forward_ms", "size"))
    attention = layers.groupby(group + ["sample"]).attention_ms.sum().groupby(group).median()
    speed["profile_attention_ms"] = attention
    speed["block_density"] = layers.groupby(group).causal_block_density.mean()
    speed = speed.reset_index()
    dense = speed[speed.method == "dense"][["rope", "seq_len", "forward_ms"]].rename(columns={"forward_ms": "dense_ms"})
    speed = speed.merge(dense, on=["rope", "seq_len"])
    speed["speedup"] = speed.dense_ms / speed.forward_ms
    speed.to_csv(folder / "speed_summary.csv", index=False)

    def errors(keys):
        return tokens.groupby(keys).agg(tokens=("delta_nll", "size"),
            dense_nll=("dense_nll", "mean"), nll=("nll", "mean"), delta_nll=("delta_nll", "mean"),
            delta_p50=("delta_nll", "median"), delta_p95=("delta_nll", lambda x: x.quantile(.95)),
            delta_p99=("delta_nll", lambda x: x.quantile(.99)), delta_max=("delta_nll", "max"),
            fraction_delta_gt_1=("delta_nll", lambda x: (x > 1).mean()),
            mean_kl=("kl", "mean"), same_top1=("same_top1", "mean")).reset_index()

    error = errors(group)
    error.to_csv(folder / "error_summary.csv", index=False)
    positions = errors(group + ["region", "tail_band"])
    positions.to_csv(folder / "position_summary.csv", index=False)
    errors(group + ["sample", "region", "tail_band"]).to_csv(folder / "per_sample_errors.csv", index=False)
    worst = tokens.sort_values("delta_nll", ascending=False).groupby(group).head(20)
    worst.to_csv(folder / "worst_tokens.csv", index=False)
    pref = prefix.groupby(group).agg(probes=("nll", "size"), dense_nll=("dense_nll", "mean"),
        nll=("nll", "mean"), delta_nll=("delta_nll", "mean"), mean_kl=("kl", "mean"),
        same_top1=("same_top1", "mean")).reset_index()
    pref.to_csv(folder / "prefix_summary.csv", index=False)

    plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for method, frame in speed[speed.rope == "yarn4"].groupby("method"):
        axes[0].plot(frame.seq_len / 1024, frame.forward_ms / 1000, "o-", label=method)
        axes[1].plot(frame.seq_len / 1024, frame.speedup, "o-", label=method)
        axes[2].plot(frame.seq_len / 1024, frame.block_density * 100, "o-", label=method)
    for ax, title in zip(axes, ["Full forward (s), no KV retention", "Full-forward speedup", "Selected causal blocks (%)"]):
        ax.set(xlabel="Input length (Ki tokens)", title=title)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(folder / "study_speed.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for method, frame in error[error.rope == "yarn4"].groupby("method"):
        for ax, column in zip(axes, ["delta_nll", "delta_p99", "fraction_delta_gt_1"]):
            ax.plot(frame.seq_len / 1024, frame[column], "o-", label=method)
    for ax, title in zip(axes, ["Mean diagnostic delta NLL", "99th percentile delta NLL", "Fraction of tokens: delta NLL > 1"]):
        ax.set(xlabel="Input length (Ki tokens)", title=title)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(folder / "study_errors.png")
    plt.close(fig)

    lines = ["# FlashPrefill：128K 位置误差与末尾保护实验", "",
        "同一模型 BF16、batch=1。主实验全长度统一 YaRN 4×；32K 另做原始 RoPE 对照。"
        "各输入由不同 gov_report 文档拼接，流内不重复文档；相同样本在各长度取同一 token 流的前缀。", "",
        "attention 处理完整输入序列。MLP/RMSNorm 按 token 分块；不保留跨层 KV cache。"
        "下表是完整模型前向加末位置 LM head/argmax 的耗时，不是含持久 KV 的生成 TTFT，也不测 decode。", "",
        "kernel 数值检查见 kernel_check.json，逐 token 算子分块检查和完整配置见各目录 metadata.json。", "",
        "## 速度与密度", "",
        "| RoPE | tokens | 方法 | 前向 ms | 加速比 | 峰值 GiB | 保留块 | attention profile ms |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in speed.itertuples():
        lines.append(f"| {r.rope} | {r.seq_len} | {r.method} | {r.forward_ms:.1f} | {r.speedup:.2f}× | "
                     f"{r.peak_gib:.2f} | {r.block_density:.1%} | {r.profile_attention_ms:.1f} |")
    lines += ["", "计时取重复测量中位数；profile 是额外前向，含事件及块计数开销，只用于定位，未混入主计时。", "",
        "## 整篇前向的逐 token 误差诊断", "",
        "块路由会读取同块后续 query。以下中间位置指标用于 Dense/稀疏表示差异诊断，不作为严格自回归 PPL。"
        "正 ΔNLL 表示目标 token 的概率下降。早/中/晚各抽取 128 tokens，文档流尾部抽取 512 tokens；排除报告分隔符。", "",
        "| RoPE | tokens | 方法 | 平均 ΔNLL | P95 | P99 | 最大值 | ΔNLL>1 比例 | 平均 KL | top1 一致率 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in error.itertuples():
        lines.append(f"| {r.rope} | {r.seq_len} | {r.method} | {r.delta_nll:+.4f} | {r.delta_p95:.4f} | "
                     f"{r.delta_p99:.4f} | {r.delta_max:.4f} | {r.fraction_delta_gt_1:.1%} | {r.mean_kl:.5f} | {r.same_top1:.1%} |")
    lines += ["", "## 位置与保护区域", "",
        "tail_band 根据预测所用 query 的位置划分，跨消融保持相同分组：last_block、penultimate_block、earlier。"
        "末尾全注意力位置仍可受历史表示变化影响。", "",
        "| RoPE | tokens | 方法 | 区域 | tail band | 平均 ΔNLL | P99 |",
        "| --- | ---: | --- | --- | --- | ---: | ---: |"]
    for r in positions.itertuples():
        lines.append(f"| {r.rope} | {r.seq_len} | {r.method} | {r.region} | {r.tail_band} | {r.delta_nll:+.4f} | {r.delta_p99:.4f} |")
    lines += ["", "## 只输入前缀的预测", "",
        "在文档流约 25%、50%、75% 位置截断输入，只预测下一个真实 token；目标与后续文本均不进入模型。"
        "每个样本三个 probe，单独给出与 Dense 的差异；这是小样本预测检查，不是全篇困惑度。", "",
        "| RoPE | tokens | 方法 | probes | Dense NLL | NLL | ΔNLL | KL | top1 一致率 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in pref.itertuples():
        lines.append(f"| {r.rope} | {r.seq_len} | {r.method} | {r.probes} | {r.dense_nll:.4f} | "
                     f"{r.nll:.4f} | {r.delta_nll:+.4f} | {r.mean_kl:.5f} | {r.same_top1:.1%} |")
    lines += ["", "![速度与密度](study_speed.png)", "", "![误差分布](study_errors.png)", "",
        "明细：position_summary.csv、per_sample_errors.csv、worst_tokens.csv、prefix_summary.csv。"
        "P95/P99 为采样 token 的分位数，不是置信区间；重复计时不增加独立文档样本数。", ""]
    (folder / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {folder / 'REPORT.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    report(parser.parse_args().folder)
