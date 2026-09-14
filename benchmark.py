"""Qwen2.5-7B: original FlashPrefill V1 versus dense Flash SDPA."""

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import subprocess
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import triton

from attention import AttentionBackend
from data import prepare_inputs


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--lengths", nargs="+", type=int, default=[4096, 8192, 16384, 32768])
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--new-tokens", type=int, default=64)
    p.add_argument("--quality-tokens", type=int, default=512)
    p.add_argument("--alpha", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("results/qwen25_7b"))
    return p.parse_args()


class Output:
    def __init__(self, folder, name, fields):
        self.file = (folder / name).open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=fields)
        self.writer.writeheader()

    def write(self, row):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


@torch.inference_mode()
def generate(model, input_ids, new_tokens):
    """One prefill gives token 1; N-1 cached forwards give the remaining tokens."""
    torch.compiler.cudagraph_mark_step_begin()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    out = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    token = out.logits[:, -1].argmax(-1, keepdim=True)
    cache = out.past_key_values
    tokens = [token]
    del out
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - started) * 1000
    prefill_peak = torch.cuda.max_memory_allocated() / 2**30

    started = time.perf_counter()
    for step in range(new_tokens - 1):
        out = model(
            input_ids=token, past_key_values=cache, use_cache=True,
            cache_position=torch.tensor([input_ids.shape[1] + step], device="cuda"),
            logits_to_keep=1,
        )
        token = out.logits[:, -1].argmax(-1, keepdim=True)
        cache = out.past_key_values
        tokens.append(token)
        del out
    torch.cuda.synchronize()
    decode_ms = (time.perf_counter() - started) * 1000
    generated = torch.cat(tokens, dim=1)[0].tolist()
    metrics = {
        "prefill_ms": prefill_ms,
        "prefill_tokens_s": input_ids.shape[1] / (prefill_ms / 1000),
        "decode_ms": decode_ms,
        "decode_steps": new_tokens - 1,
        "decode_ms_token": decode_ms / (new_tokens - 1),
        "decode_tokens_s": (new_tokens - 1) / (decode_ms / 1000),
        "total_ms": prefill_ms + decode_ms,
        "prefill_peak_gib": prefill_peak,
        "generation_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    del cache
    return metrics, generated


@torch.inference_mode()
def quality(model, backend, input_ids, sample, count):
    """Teacher-forced NLL on the last document tokens; no full-vocabulary N*V tensor."""
    torch.compiler.cudagraph_mark_step_begin()
    backend.start_profile()
    hidden = model.model(input_ids=input_ids, use_cache=False).last_hidden_state
    profile = backend.finish_profile()
    end = sample["document_end"]
    begin = end - count
    loss = torch.zeros((), device="cuda")
    for left in range(begin, end, 64):
        right = min(left + 64, end)
        logits = model.lm_head(hidden[:, left - 1:right - 1]).float()
        loss += F.cross_entropy(logits[0], input_ids[0, left:right], reduction="sum")
        del logits
    final_log_probs = model.lm_head(hidden[:, -1]).float().log_softmax(-1)[0].cpu()
    nll = loss.item() / count
    del hidden
    return {"tail_nll": nll, "tail_ppl": math.exp(nll)}, final_log_probs, profile


def main():
    args = arguments()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    inputs, data_revision = prepare_inputs(tokenizer, args.lengths, args.samples, args.seed)
    torch.save(inputs, args.out / "inputs.pt")
    with (args.out / "inputs.jsonl").open("w", encoding="utf-8") as f:
        for item in inputs:
            f.write(json.dumps({k: v for k, v in item.items() if k != "input_ids"}) + "\n")

    backend = AttentionBackend(args.alpha)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda",
    ).eval()
    metadata = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model_commit": model.config._commit_hash,
        "model_config": model.config.to_dict(),
        "data_revision": data_revision,
        "upstream_commit": "baa612047433a992a00d07dc178205eed065ae14",
        "exp2_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "triton": triton.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "gpu_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "nvidia_smi": subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader",
        ], text=True).strip(),
        "attention": {"block_size": 128, "alpha": args.alpha, "sink_blocks": 2,
                      "window_blocks": 4, "last_query_blocks_full": 2,
                      "min_budget": 0, "sparse_layers": "all 28", "decode": "dense Flash SDPA"},
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    common = ["method", "seq_len", "sample", "source_id"]
    timings = Output(args.out, "timings.csv", common + [
        "repeat", "prefill_ms", "prefill_tokens_s", "decode_ms", "decode_steps",
        "decode_ms_token", "decode_tokens_s", "total_ms", "prefill_peak_gib",
        "generation_peak_gib", "peak_reserved_gib",
    ])
    qualities = Output(args.out, "quality.csv", common + [
        "quality_tokens", "tail_nll", "tail_ppl", "next_token_kl_dense_to_method",
        "same_next_token_as_dense",
    ])
    layers = Output(args.out, "layers.csv", common + ["layer", "attention_ms", "causal_block_density"])
    generations = (args.out / "generations.jsonl").open("w", encoding="utf-8")

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for length in args.lengths:
            batch = [item for item in inputs if item["seq_len"] == length]
            warm_ids = torch.tensor([batch[0]["input_ids"]], device="cuda")
            for method in ("dense", "flashprefill"):
                backend.method = method
                print(f"Warmup {method} {length}: compile/autotune excluded", flush=True)
                generate(model, warm_ids, args.new_tokens)
            del warm_ids

            for sample in batch:
                ids = torch.tensor([sample["input_ids"]], device="cuda")
                for repeat in range(args.repeats):
                    # Alternate execution order to reduce temperature/clock order effects.
                    methods = ("dense", "flashprefill") if repeat % 2 == 0 else ("flashprefill", "dense")
                    for method in methods:
                        backend.method = method
                        metrics, generated = generate(model, ids, args.new_tokens)
                        key = {"method": method, "seq_len": length, "sample": sample["sample"],
                               "source_id": sample["source_id"]}
                        timings.write({**key, "repeat": repeat, **metrics})
                        generations.write(json.dumps({**key, "repeat": repeat, "token_ids": generated,
                                                       "text": tokenizer.decode(generated)}, ensure_ascii=False) + "\n")
                        generations.flush()
                        print(f"{method:12s} n={length:5d} sample={sample['sample']} rep={repeat} "
                              f"prefill={metrics['prefill_ms']:.1f}ms "
                              f"decode={metrics['decode_tokens_s']:.1f}tok/s "
                              f"peak={metrics['generation_peak_gib']:.2f}GiB", flush=True)

                reference = None
                for method in ("dense", "flashprefill"):
                    backend.method = method
                    result, log_probs, profile = quality(model, backend, ids, sample, args.quality_tokens)
                    if method == "dense":
                        reference = log_probs
                    key = {"method": method, "seq_len": length, "sample": sample["sample"],
                           "source_id": sample["source_id"]}
                    qualities.write({**key, "quality_tokens": args.quality_tokens, **result,
                                     "next_token_kl_dense_to_method": (reference.exp() * (reference - log_probs)).sum().item(),
                                     "same_next_token_as_dense": int(reference.argmax() == log_probs.argmax())})
                    for row in profile:
                        layers.write({**key, **row})
                    print(f"Quality {method} {length}: tail NLL={result['tail_nll']:.4f}", flush=True)
                del ids
            torch.cuda.empty_cache()

    for output in (timings, qualities, layers):
        output.close()
    generations.close()
    from report import make_report
    make_report(args.out)
    print(f"Completed: {args.out / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
