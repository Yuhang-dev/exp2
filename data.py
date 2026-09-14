"""Reuse exp1's LongBench cache; identical documents across lengths/backends."""

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from huggingface_hub import hf_hub_download


def prepare_inputs(tokenizer, lengths, samples, seed):
    archive = hf_hub_download("zai-org/LongBench", "data.zip", repo_type="dataset")
    with zipfile.ZipFile(archive) as z:
        rows = [json.loads(line) for line in z.read("data/gov_report.jsonl").splitlines()]

    template = tokenizer.apply_chat_template(
        [{"role": "user", "content":
          "Read the following document.\n\n<<<DOCUMENT>>>\n\nSummarize this document."}],
        tokenize=False, add_generation_prompt=True,
    )
    prefix, suffix = template.split("<<<DOCUMENT>>>")
    left = tokenizer.encode(prefix, add_special_tokens=False)
    right = tokenizer.encode(suffix, add_special_tokens=False)
    maximum = max(lengths) - len(left) - len(right)
    documents = []
    for index in np.random.default_rng(seed).permutation(len(rows)):
        row = rows[index]
        context = tokenizer.encode(row["context"], add_special_tokens=False)
        if len(context) >= maximum:
            documents.append((row["_id"], context))
        if len(documents) == samples:
            break
    if len(documents) != samples:
        raise ValueError(f"gov_report has only {len(documents)} documents long enough for this run")

    inputs = []
    for length in lengths:
        for sample, (source_id, context) in enumerate(documents):
            budget = length - len(left) - len(right)
            ids = left + context[:budget] + right
            inputs.append({
                "seq_len": length, "sample": sample, "source_id": source_id,
                "source_context_tokens": len(context), "input_ids": ids,
                "document_start": len(left), "document_end": len(left) + budget,
                "input_sha256": hashlib.sha256(np.asarray(ids, dtype=np.int32).tobytes()).hexdigest(),
            })
    return inputs, Path(archive).parent.name
