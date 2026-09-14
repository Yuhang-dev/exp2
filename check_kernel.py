"""GPU numerical audit: independent FP32 references for the Qwen GQA kernels."""

import argparse
import json
from pathlib import Path

import torch
import triton
from triton.runtime.driver import driver

from upstream import flashprefill_native_forward as ops


def check(name, actual, expected, records, atol=.02, rtol=.02):
    actual, expected = actual.float(), expected.float()
    delta = actual - expected
    relative = delta.norm() / expected.norm().clamp_min(1e-12)
    record = {"check": name, "max_abs": delta.abs().max().item(),
              "relative_l2": relative.item(), "atol": atol, "rtol": rtol}
    records.append(record)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    assert relative < .02, record
    print(f"PASS {name}: max_abs={record['max_abs']:.6g}, rel_L2={relative:.6g}", flush=True)


def select_reference(score, last_full, alpha=.08):
    batch, blocks, _, heads = score.shape
    q = torch.arange(blocks, device=score.device).view(1, blocks, 1, 1)
    k = torch.arange(blocks, device=score.device).view(1, 1, blocks, 1)
    keep = ((score >= score.amax(2, keepdim=True) * alpha) | (k < 2)
            | ((q - k >= 0) & (q - k < 4)) | (q >= blocks - last_full)) & (k <= q)
    indices = torch.where(keep, k.expand(batch, blocks, blocks, heads), blocks).sort(2).values
    return indices, keep.sum(2).int()


def score_reference(q, mean_k):
    batch, length, heads, dim = q.shape
    blocks = mean_k.shape[1]
    k = mean_k.repeat_interleave(heads // mean_k.shape[2], dim=2)
    logits = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / dim**.5
    allowed = torch.arange(length, device=q.device)[:, None] >= (
        (torch.arange(blocks, device=q.device)[None, :] + 1) * 128 - 1)
    logits.masked_fill_(~allowed, -torch.inf)
    logits = torch.nn.functional.pad(logits, (0, 0, 0, blocks * 128 - length), value=-torch.inf)
    logits = logits.reshape(batch, heads, blocks, 128, blocks)
    maximum = logits.amax((3, 4), keepdim=True)
    maximum = torch.where(torch.isfinite(maximum), maximum, 0)
    mass = (logits - maximum).exp().sum(3)
    score = mass / (mass.sum(-1, keepdim=True) + 1e-9)
    return score.permute(0, 2, 3, 1).contiguous()


def prepare_score_launch(q, k):
    batch, length, heads, dim = q.shape
    kv_heads = k.shape[2]
    blocks = triton.cdiv(length, 128)
    mean = torch.empty(batch, blocks, kv_heads, dim, dtype=k.dtype, device=k.device)
    ops.compute_mean_vector[(blocks, batch * kv_heads, 1)](
        k, mean, *k.stride(), *mean.stride(), kv_heads, length, 128, dim)
    score = torch.full((batch, blocks, blocks, heads), -torch.inf, device=q.device)
    maximum = torch.full_like(score, -torch.inf)
    args = (
        q, mean, dim**-.5, score, maximum, *q.stride(), *mean.stride(),
        *score.stride(), *maximum.stride(), heads, kv_heads, length, blocks,
        128)
    return mean, score, maximum, args, (blocks, batch * heads, 1)


def raw_scores(q, k):
    mean, score, maximum, args, grid = prepare_score_launch(q, k)
    ops.compute_block_score[grid](*args, K_STRIDE=128, D_HEAD=q.shape[-1])
    return mean, score, maximum


def score_kernel(q, k):
    mean, score, maximum = raw_scores(q, k)
    return mean, ops.normalize_scores(score, maximum).clone()


def zero_score_counts(length):
    blocks = triton.cdiv(length, 128)
    return [[max(0, min((qb + 1) * 128, length) - max(qb * 128, (kb + 1) * 128 - 1))
             for kb in range(blocks)] for qb in range(blocks)]


def check_zero_scores(q, k, records):
    q = torch.zeros_like(q)
    counts = torch.tensor(zero_score_counts(q.shape[1]), dtype=torch.float32, device=q.device)
    expected = counts[None, :, :, None].expand(1, -1, -1, 28)
    expected_max = torch.where(expected > 0, 0.0, -torch.inf)
    _, raw, maximum, args, grid = prepare_score_launch(q, k)
    launch = ops.compute_block_score.fn
    shared_limit = driver.active.utils.get_device_properties(
        driver.active.get_current_device())["max_shared_mem"]
    checked, excluded = [], []
    for config in ops.get_score_configs():
        options = {"K_STRIDE": 128, "D_HEAD": q.shape[-1], **config.all_kwargs()}
        # JIT warmup compiles without loading or launching the CUDA kernel.
        compiled = launch.warmup(*args, grid=grid, **options)
        resource = {"config": config.all_kwargs(), "shared_bytes": compiled.metadata.shared}
        if compiled.metadata.shared > shared_limit:
            excluded.append(resource)
            print(f"EXCLUDED {config}: shared memory {compiled.metadata.shared} > {shared_limit} bytes", flush=True)
            continue
        raw.fill_(-torch.inf)
        maximum.fill_(-torch.inf)
        launch[grid](*args, **options)
        label = f"zero_q_{q.shape[1]} {config}"
        torch.testing.assert_close(raw, expected, atol=0, rtol=0, msg=lambda message: f"{label}\n{message}")
        torch.testing.assert_close(maximum, expected_max, atol=0, rtol=0, msg=lambda message: f"{label}\n{message}")
        checked.append(resource)
    assert checked, "No score configuration fits the GPU shared memory limit"
    records.append({"check": f"zero_q_raw_scores_{q.shape[1]}",
                    "configs_checked": len(checked), "exact_match": True,
                    "shared_memory_limit_bytes": shared_limit, "checked_configs": checked,
                    "excluded_shared_memory": excluded})
    print(f"PASS zero_q_raw_scores_{q.shape[1]}: {len(checked)} configs checked, "
          f"{len(excluded)} excluded by shared memory limit", flush=True)


def sampled_score_reference(q, mean_k, query_blocks):
    # Q @ K_mean.T with FP32 accumulation; materialize only sampled query blocks.
    k = mean_k.repeat_interleave(q.shape[2] // mean_k.shape[2], dim=2).float()
    key_ends = (torch.arange(mean_k.shape[1], device=q.device) + 1) * 128 - 1
    results = []
    for block in query_blocks:
        left, right = block * 128, min((block + 1) * 128, q.shape[1])
        logits = torch.einsum("bqhd,bkhd->bhqk", q[:, left:right].float(), k) / q.shape[-1]**.5
        allowed = torch.arange(left, right, device=q.device)[:, None] >= key_ends[None, :]
        logits.masked_fill_(~allowed, -torch.inf)
        maximum = logits.amax((2, 3), keepdim=True)
        maximum = torch.where(torch.isfinite(maximum), maximum, 0)
        mass = (logits - maximum).exp().sum(2)
        results.append(mass / (mass.sum(-1, keepdim=True) + 1e-9))
    return torch.stack(results, dim=2).permute(0, 2, 3, 1).contiguous()


def attention_kernel(q, k, v, indices, counts):
    batch, length, heads, dim = q.shape
    out = torch.empty_like(q)
    ops._flash_forward[lambda meta: (triton.cdiv(length, meta["Q_TILE_SIZE"]), batch * heads, 1)](
        q, k, v, out, indices, counts, dim**-.5,
        *q.stride(), *k.stride(), *v.stride(), *out.stride(),
        *indices.stride(), *counts.stride(), length, length, heads, k.shape[2],
        BLOCK_SIZE=128, D_HEAD=dim)
    return out


def attention_reference(q, k, v, indices, counts, positions=None):
    batch, length, heads, dim = q.shape
    blocks = indices.shape[1]
    if positions is None:
        positions = torch.arange(length, device=q.device)
    block_mask = torch.zeros(batch, blocks, blocks + 1, heads, dtype=torch.bool, device=q.device)
    valid = torch.arange(blocks, device=q.device).view(1, 1, blocks, 1) < counts.unsqueeze(2)
    block_mask.scatter_(2, indices, valid)
    block_mask = block_mask[:, :, :blocks].permute(0, 3, 1, 2)
    allowed = block_mask[:, :, positions // 128].repeat_interleave(128, -1)[..., :length]
    allowed &= positions[:, None] >= torch.arange(length, device=q.device)[None, :]
    k = k.repeat_interleave(heads // k.shape[2], 2).float()
    v = v.repeat_interleave(heads // v.shape[2], 2).float()
    logits = torch.einsum("bqhd,bkhd->bhqk", q[:, positions].float(), k) / dim**.5
    logits.masked_fill_(~allowed, -torch.inf)
    return torch.einsum("bhqk,bkhd->bqhd", logits.softmax(-1), v)


def fixed_indices(length):
    blocks = triton.cdiv(length, 128)
    q = torch.arange(blocks, device="cuda").view(1, blocks, 1, 1)
    k = torch.arange(blocks, device="cuda").view(1, 1, blocks, 1)
    h = torch.arange(28, device="cuda").view(1, 1, 1, 28)
    # Different sparse patterns across GQA heads; diagonal always present.
    keep = (k == q) | (k == 0) | (k == (q + h) % (q + 1))
    indices = torch.where(keep, k, blocks).sort(2).values.contiguous()
    return indices, keep.sum(2).int()


def query_dependency(records):
    """A deterministic structural example, not a failed attention arithmetic test."""
    q = torch.zeros(1, 1024, 28, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(1, 1024, 4, 128, device="cuda", dtype=torch.bfloat16)
    k[:, 256:384, :, 0] = 1
    k[:, 384:512, :, 1] = 1
    q[:, 896:960, :, 0] = 4
    changed = q.clone()
    changed[:, 960:, :, 1] = 128
    _, first = score_kernel(q, k)
    _, second = score_kernel(changed, k)
    a, _ = select_reference(first, 0)
    b, _ = select_reference(second, 0)
    different = int((a[:, 7] != b[:, 7]).sum().item())
    assert different > 0, "Constructed query-dependency example did not change selection"
    records.append({"check": "within_block_later_queries_change_routing", "different_indices": different,
                    "interpretation": "full-pass interior NLL is diagnostic, not strict autoregressive PPL"})
    print(f"OBSERVED within-block later-query dependence: {different} changed indices", flush=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/kernel_check.json"))
    args = parser.parse_args()
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    records = []
    for length in (1024, 1089):
        torch.compiler.cudagraph_mark_step_begin()
        q = torch.randn(1, length, 28, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, length, 4, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        print(f"Checking raw zero-Q scores at {length} for every autotune config", flush=True)
        check_zero_scores(q, k, records)
        mean, score = score_kernel(q, k)
        mean_ref = torch.stack([k[:, p:p + 128].float().mean(1) for p in range(0, length, 128)], 1)
        check(f"mean_{length}", mean, mean_ref, records, atol=.002, rtol=.02)
        ref_score = score_reference(q, mean)
        check(f"proxy_score_{length}", score, ref_score, records, atol=2e-4, rtol=.005)

        # Non-uniform scores exercise sparsity and all three tail-protection settings.
        score_input = score.square().square()
        for last in (0, 1, 2):
            indices, counts = ops.deal_output_score(score_input, 2, 4, .8, last, 0)
            ri, rc = select_reference(score_input, last, .8)
            torch.testing.assert_close(indices, ri, atol=0, rtol=0)
            torch.testing.assert_close(counts, rc, atol=0, rtol=0)
            records.append({"check": f"selection_{length}_tail{last}", "exact_match": True})

        indices, counts = fixed_indices(length)
        sparse = attention_kernel(q, k, v, indices, counts)
        check(f"same_mask_attention_{length}", sparse,
              attention_reference(q, k, v, indices, counts), records)
        cut = length - 70
        k2, v2 = k.clone(), v.clone()
        k2[:, cut:] *= 20
        v2[:, cut:] += 100
        changed = attention_kernel(q, k2, v2, indices, counts)
        check(f"fixed_mask_causal_{length}", changed[:, :cut], sparse[:, :cut], records,
              atol=0, rtol=0)

        full = torch.empty_like(q)
        ops.flash_prefill(q, k, v, full, 128, 2, 4, 0.0, 0, 0)
        dense_indices, dense_counts = select_reference(score, 0, 0.0)
        check(f"all_keep_dense_{length}", full,
              attention_reference(q, k, v, dense_indices, dense_counts), records)
        sparse = torch.empty_like(q)
        ops.flash_prefill(q, k, v, sparse, 128, 2, 4, .8, 0, 0)
        ri, rc = select_reference(ref_score, 0, .8)
        check(f"end_to_end_{length}", sparse, attention_reference(q, k, v, ri, rc), records)
        del q, k, v, k2, v2, full, sparse, changed

    query_dependency(records)
    torch.compiler.cudagraph_mark_step_begin()
    length = 131072
    q = torch.randn(1, length, 28, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, length, 4, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    indices, counts = fixed_indices(length)
    out = attention_kernel(q, k, v, indices, counts)
    positions = torch.tensor([0, 127, 128, 32767, 32768, 65535, 65536, 131070, 131071], device="cuda")
    check("128k_indexing_and_boundaries", out[:, positions],
          attention_reference(q, k, v, indices, counts, positions), records)
    mean, score = score_kernel(q, k)
    query_blocks = [0, 1, 255, 256, 511, 512, 1023]
    check("128k_proxy_score_boundaries", score[:, query_blocks],
          sampled_score_reference(q, mean, query_blocks), records, atol=2e-4, rtol=.005)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"status": "PASS", "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__, "triton": triton.__version__,
        "score_implementation": ops.SCORE_IMPL, "checks": records}, indent=2), encoding="utf-8")
    print(f"Kernel audit passed: {args.out}", flush=True)


if __name__ == "__main__":
    main()
