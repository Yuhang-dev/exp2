"""Aggregate supplied error CSVs and select concrete cases for prefix replay."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def read_csv(path, integers, floats):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in integers:
            row[key] = int(row[key])
        for key in floats:
            row[key] = float(row[key])
    return rows


def aggregate(rows):
    tokens = sum(row["tokens"] for row in rows)
    result = {"tokens": tokens}
    for key in ("delta_nll", "mean_kl", "same_top1"):
        result[key] = sum(row[key] * row["tokens"] for row in rows) / tokens
    result["delta_gt_1_count"] = sum(round(row["fraction_delta_gt_1"] * row["tokens"]) for row in rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, nargs="?", default=Path("results/study_4090"))
    args = parser.parse_args()
    worst = read_csv(args.folder / "worst_tokens.csv",
                     ("seq_len", "sample", "position", "token_id", "tail_protected", "same_top1"),
                     ("dense_nll", "nll", "delta_nll", "kl"))
    samples = read_csv(args.folder / "per_sample_errors.csv", ("seq_len", "sample", "tokens"),
                       ("delta_nll", "mean_kl", "same_top1", "fraction_delta_gt_1"))
    default = [row for row in samples if row["method"] == "flashprefill_tail2"]
    groups = defaultdict(list)
    for row in default:
        groups[(row["rope"], row["seq_len"], row["sample"])].append(row)
    summaries = [{"rope": rope, "seq_len": length, "sample": sample, **aggregate(rows)}
                 for (rope, length, sample), rows in sorted(groups.items())]

    rope_pairs = []
    for sample in range(4):
        native = aggregate(groups[("native", 32768, sample)])
        yarn = aggregate(groups[("yarn4", 32768, sample)])
        rope_pairs.append({"sample": sample, "native": native, "yarn4": yarn,
                           "kl_ratio_yarn_to_native": yarn["mean_kl"] / native["mean_kl"]})

    selected = [
        ("64k_shared_query_block", "yarn4", 65536, 2, 58924),
        ("16k_protected_fcc", "yarn4", 16384, 3, 16228),
        ("16k_table_chron", "yarn4", 16384, 3, 14709),
        ("128k_early_figure", "yarn4", 131072, 0, 13079),
        ("128k_late_month", "yarn4", 131072, 0, 117918),
        ("32k_native_e", "native", 32768, 3, 32284),
        ("32k_yarn_e", "yarn4", 32768, 3, 32284),
    ]
    cases = []
    for name, rope, length, sample, position in selected:
        matches = [row for row in worst if (row["rope"], row["seq_len"], row["sample"], row["position"])
                   == (rope, length, sample, position)]
        first = matches[0]
        full_blocks, prefix_blocks = (length + 127) // 128, (position + 127) // 128
        variants = []
        for row in sorted(matches, key=lambda row: row["method"]):
            last_full = int(row["method"][-1])
            variants.append({key: row[key] for key in
                             ("method", "tail_protected", "dense_nll", "nll", "delta_nll", "kl", "same_top1")})
            variants[-1].update(
                dense_target_probability=math.exp(-row["dense_nll"]),
                sparse_target_probability=math.exp(-row["nll"]),
                # Truncate before the target; keep the original absolute guard boundary.
                prefix_last_full_preserving_original=max(0, prefix_blocks - (full_blocks - last_full)),
            )
        cases.append({"case": name, "rope": rope, "full_input_length": length,
                      "sample": sample, "target_position": position, "query_position": position - 1,
                      "query_block": (position - 1) // 128, "prefix_length": position,
                      "source_id": first["source_id"], "token_id": first["token_id"],
                      "token": first["token"], "context_before": first["context_before"],
                      "variants": variants})

    at_64k = [row for row in default if row["seq_len"] == 65536]
    late = [row for row in at_64k if row["region"] == "late"]
    cluster = [row for row in worst if row["method"] == "flashprefill_tail2"
               and row["seq_len"] == 65536 and row["sample"] == 2 and row["region"] == "late"]
    output = {
        "sources": ["worst_tokens.csv", "per_sample_errors.csv"],
        "worst_rows": len(worst), "per_sample_rows": len(samples),
        "distinct_locations_across_methods": len({(r["rope"], r["seq_len"], r["sample"], r["position"])
                                                    for r in worst}),
        "method": "Token-count-weighted means; recover threshold counts with round(fraction*tokens). "
                  "Do not average subgroup quantiles. Worst rows are top-20 selections, not a full token population.",
        "tail2_per_sample": summaries, "rope_pairs_32k_tail2": rope_pairs,
        "cluster_64k": {
            "all_samples": aggregate(at_64k), "late_all_samples": aggregate(late),
            "late_excluding_sample2": aggregate([r for r in late if r["sample"] != 2]),
            "all_excluding_sample2_late": aggregate([r for r in at_64k
                                                    if not (r["sample"] == 2 and r["region"] == "late")]),
            "selected_tokens": [{"position": r["position"], "query_block": (r["position"] - 1) // 128,
                                 "token": r["token"], "delta_nll": r["delta_nll"],
                                 "dense_probability": math.exp(-r["dense_nll"]),
                                 "sparse_probability": math.exp(-r["nll"])} for r in cluster],
        },
        "replay_cases": cases,
        "replay_status": "Selected from full-pass diagnostics; strict-prefix replay and layer attribution not run.",
    }
    destination = args.folder / "failure_analysis.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Analyzed {len(worst)} worst-token rows and {len(samples)} sample/region rows: {destination}")


if __name__ == "__main__":
    main()
