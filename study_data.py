"""Fixed, non-repeating report streams reused at every length and RoPE setting."""

import hashlib
import json
from pathlib import Path
import zipfile

from huggingface_hub import hf_hub_download
import numpy as np
import torch


def prepare(tokenizer, folder, samples, seed):
    archive = hf_hub_download("zai-org/LongBench", "data.zip", repo_type="dataset")
    with zipfile.ZipFile(archive) as z:
        rows = [json.loads(line) for line in z.read("data/gov_report.jsonl").splitlines()]
    template = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Read the following reports.\n\n<<<REPORTS>>>\n\nSummarize these reports."}],
        tokenize=False, add_generation_prompt=True,
    )
    before, after = template.split("<<<REPORTS>>>")
    left = tokenizer.encode(before, add_special_tokens=False)
    right = tokenizer.encode(after, add_special_tokens=False)
    budget = 131072 - len(left) - len(right)
    streams = []
    for sample in range(samples):
        ids, segments = [], []
        for index in np.random.default_rng(seed + sample).permutation(len(rows)):
            row = rows[index]
            marker = tokenizer.encode(f"\n\n[Report {len(segments) + 1}]\n\n", add_special_tokens=False)
            ids.extend(marker)
            begin = len(ids)
            ids.extend(tokenizer.encode(row["context"], add_special_tokens=False))
            segments.append({"source_id": row["_id"], "start": begin, "end": len(ids)})
            if len(ids) >= budget:
                break
        assert len(ids) >= budget, "gov_report corpus is shorter than the requested stream"
        streams.append({"sample": sample, "tokens": ids[:budget], "segments": segments})
    data = {"prefix": left, "suffix": right, "streams": streams,
            "data_revision": Path(archive).parent.name, "seed": seed}
    torch.save(data, folder / "streams.pt")
    return data


def at_length(data, sample, length):
    stream = data["streams"][sample]
    start = len(data["prefix"])
    end = length - len(data["suffix"])
    ids = data["prefix"] + stream["tokens"][:end - start] + data["suffix"]
    return ids, {"sample": sample, "seq_len": length, "document_start": start,
                 "document_end": end,
                 "sha256": hashlib.sha256(np.asarray(ids, dtype=np.int32).tobytes()).hexdigest(),
                 "segments": [{**s, "start": s["start"] + start, "end": min(s["end"] + start, end)}
                              for s in stream["segments"] if s["start"] + start < end]}


def windows(info, width=128):
    start, end = info["document_start"], info["document_end"]
    positions = {}
    for label, fraction in [("early", .1), ("middle", .5), ("late", .9)]:
        left = int(start + (end - start) * fraction) - width // 2
        positions.update((p, label) for p in range(left, left + width))
    positions.update((p, "tail") for p in range(end - 512, end))
    # Skip report separators; a target and its preceding token must be in a report.
    return [(p, label) for p, label in sorted(positions.items())
            if any(s["start"] < p < s["end"] for s in info["segments"])]


def prefix_targets(info):
    start, end = info["document_start"], info["document_end"]
    candidates = [int(start + (end - start) * f) for f in (.25, .5, .75)]
    result = []
    for target in candidates:
        # Inside a block: probes also exercise partial-block scoring.
        target = target // 128 * 128 + 65
        segment = next(s for s in info["segments"] if s["end"] > target)
        result.append(max(target, segment["start"] + 1))
    return result
