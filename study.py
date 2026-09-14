"""Position errors, tail-protection ablations and strict prefix readouts up to 128K."""

import argparse
import json
from pathlib import Path
import subprocess
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer
import transformers
import triton
from upstream import flashprefill_native_forward as ops

from attention import AttentionBackend
from benchmark import Output
from check_kernel import check
from study_data import at_length, prefix_targets, windows
from study_model import hidden_forward, load_model


METHODS = ("dense", "flashprefill_tail0", "flashprefill_tail1", "flashprefill_tail2")


def configure(backend, method):
    backend.method = "dense" if method == "dense" else "flashprefill"
    backend.last_full = 0 if method == "dense" else int(method[-1])


@torch.inference_mode()
def check_chunking(model):
    hidden = torch.randn(1, 2049, 3584, dtype=torch.bfloat16, device="cuda")
    records = []
    mlp = model.model.layers[0].mlp
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    # FP32 checks the split/concatenation; BF16 measures drift after reshaping GEMMs.
    # BF16 -> FP32 -> BF16 preserves the stored model weights exactly.
    for dtype, limit in [(torch.float32, 1e-5), (torch.bfloat16, .01)]:
        mlp.to(dtype=dtype)
        x = hidden.to(dtype)
        actual = mlp(x).float()
        expected = mlp.unchunked_forward(x).float()
        delta = actual - expected
        token_error = delta.norm(dim=-1) / expected.norm(dim=-1).clamp_min(1e-12)
        record = {"check": f"chunked_mlp_{dtype}", "max_abs": delta.abs().max().item(),
                  "relative_l2": (delta.norm() / expected.norm()).item(),
                  "max_token_relative_l2": token_error.max().item(), "limit": limit}
        records.append(record)
        print(json.dumps(record), flush=True)
        assert record["max_token_relative_l2"] < limit, record
        print(f"PASS {record['check']}", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = original_tf32
    norm = model.model.layers[0].input_layernorm
    check("chunked_rmsnorm", norm(hidden), norm.unchunked_forward(hidden), records)
    return records


@torch.inference_mode()
def timed(model, ids):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    hidden = hidden_forward(model, ids)
    token = model.lm_head(hidden[:, -1]).argmax(-1)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) * 1000
    peak = torch.cuda.max_memory_allocated() / 2**30
    del hidden, token
    return {"forward_ms": elapsed, "tokens_s": ids.shape[1] * 1000 / elapsed, "peak_gib": peak}


@torch.inference_mode()
def sampled_log_probs(model, backend, ids, positions):
    backend.start_profile()
    hidden = hidden_forward(model, ids)
    profile = backend.finish_profile()
    log_probs = []
    for left in range(0, len(positions), 32):
        queries = torch.tensor([p - 1 for p in positions[left:left + 32]], device="cuda")
        logits = model.lm_head(hidden[0, queries]).float()
        log_probs.append(logits.log_softmax(-1).cpu())
    del hidden
    return torch.cat(log_probs), profile


@torch.inference_mode()
def prefix_log_probs(model, ids):
    hidden = hidden_forward(model, ids)
    result = model.lm_head(hidden[:, -1]).float().log_softmax(-1)[0].cpu()
    del hidden
    return result


def token_rows(reference, current, ids, windows_, info, method, tokenizer):
    positions = [p for p, _ in windows_]
    targets = torch.tensor([ids[p] for p in positions])
    row_index = torch.arange(len(positions))
    dense_nll = -reference[row_index, targets]
    nll = -current[row_index, targets]
    # CPU batches keep memory bounded; timing is measured in separate forwards.
    kls = torch.cat([(reference[p:p + 32].exp() * (reference[p:p + 32] - current[p:p + 32])).sum(-1)
                     for p in range(0, len(positions), 32)])
    agree = reference.argmax(-1) == current.argmax(-1)
    blocks = (info["seq_len"] + 127) // 128
    last_full = int(method[-1])
    for index, (target, region) in enumerate(windows_):
        from_end = blocks - 1 - (target - 1) // 128
        band = "last_block" if from_end == 0 else "penultimate_block" if from_end == 1 else "earlier"
        source = next(s["source_id"] for s in info["segments"] if s["start"] <= target < s["end"])
        yield {"method": method, "sample": info["sample"], "seq_len": info["seq_len"],
               "position": target, "region": region, "tail_band": band,
               "tail_protected": int(from_end < last_full), "source_id": source,
               "token_id": ids[target], "token": tokenizer.decode([ids[target]]),
               "context_before": tokenizer.decode(ids[max(0, target - 24):target]),
               "dense_nll": dense_nll[index].item(), "nll": nll[index].item(),
               "delta_nll": (nll[index] - dense_nll[index]).item(), "kl": kls[index].item(),
               "same_top1": int(agree[index])}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--rope", choices=["native", "yarn4"], default="yarn4")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--token-chunk", type=int, default=1024)
    parser.add_argument("--alpha", type=float, default=.08)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main():
    args = arguments()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    data = torch.load(args.inputs, weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    backend = AttentionBackend(args.alpha)
    model = load_model(args.model, args.rope, args.token_chunk)
    checks = check_chunking(model)
    metadata = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "torch": torch.__version__, "triton": triton.__version__,
                "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(),
                "model_config": model.config.to_dict(), "model_commit": model.config._commit_hash,
                "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "data_revision": data["data_revision"], "seed": data["seed"],
                "samples": len(data["streams"]), "chunk_checks": checks,
                "measurement": "full model forward, use_cache=False, final-position LM head and argmax",
                "attention": {"alpha": args.alpha, "block_size": 128, "sink": 2, "window": 4,
                              "last_full": [0, 1, 2], "layers": 28, "score_kernel": ops.SCORE_IMPL},
                "quality": "Full-pass interior NLL/KL is diagnostic; prefix readouts exclude target/suffix."}
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    key = ["method", "sample", "seq_len"]
    timings = Output(args.out, "timings.csv", key + ["repeat", "forward_ms", "tokens_s", "peak_gib"])
    tokens = Output(args.out, "tokens.csv", key + ["position", "region", "tail_band", "tail_protected",
        "source_id", "token_id", "token", "context_before", "dense_nll", "nll", "delta_nll", "kl", "same_top1"])
    prefixes = Output(args.out, "prefix.csv", key + ["prefix_length", "token_id", "dense_nll", "nll",
                                                 "delta_nll", "kl", "same_top1"])
    layers = Output(args.out, "layers.csv", key + ["layer", "attention_ms", "causal_block_density"])
    inputs = (args.out / "inputs.jsonl").open("w", encoding="utf-8")
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for sample in range(len(data["streams"])):
            raw, info = at_length(data, sample, args.length)
            inputs.write(json.dumps(info) + "\n")
            inputs.flush()
            ids = torch.tensor([raw], device="cuda")
            if sample == 0:
                for method in METHODS:
                    configure(backend, method)
                    print(f"Warmup {args.rope} {args.length} {method}", flush=True)
                    timed(model, ids)
            for repeat in range(args.repeats):
                offset = (sample + repeat) % len(METHODS)
                for method in METHODS[offset:] + METHODS[:offset]:
                    configure(backend, method)
                    result = timed(model, ids)
                    timings.write({"method": method, "sample": sample, "seq_len": args.length,
                                   "repeat": repeat, **result})
                    print(f"{method} n={args.length} sample={sample} rep={repeat} "
                          f"forward={result['forward_ms']:.1f}ms peak={result['peak_gib']:.2f}GiB", flush=True)

            locations = windows(info)
            positions = [p for p, _ in locations]
            reference = None
            for method in METHODS:
                configure(backend, method)
                current, profile = sampled_log_probs(model, backend, ids, positions)
                if method == "dense":
                    reference = current
                else:
                    for row in token_rows(reference, current, raw, locations, info, method, tokenizer):
                        tokens.write(row)
                for row in profile:
                    layers.write({"method": method, "sample": sample, "seq_len": args.length, **row})
                print(f"Position quality {method} n={args.length} sample={sample}: {len(positions)} tokens", flush=True)
                del current
            del reference

            for target in prefix_targets(info):
                reference = None
                for method in METHODS:
                    configure(backend, method)
                    log_probs = prefix_log_probs(model, ids[:, :target])
                    if method == "dense":
                        reference = log_probs
                    prefixes.write({"method": method, "sample": sample, "seq_len": args.length,
                        "prefix_length": target, "token_id": raw[target],
                        "dense_nll": -reference[raw[target]].item(), "nll": -log_probs[raw[target]].item(),
                        "delta_nll": (reference[raw[target]] - log_probs[raw[target]]).item(),
                        "kl": (reference.exp() * (reference - log_probs)).sum().item(),
                        "same_top1": int(reference.argmax() == log_probs.argmax())})
                print(f"Prefix readout n={args.length} sample={sample} prefix={target}", flush=True)
            del ids
    inputs.close()
    for output in (timings, tokens, prefixes, layers):
        output.close()
    (args.out / "completed.json").write_text(json.dumps({"complete": True}), encoding="utf-8")
    print(f"Completed {args.out}", flush=True)


if __name__ == "__main__":
    main()
