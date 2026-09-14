"""Isolate block scoring, eager normalization, and compiled normalization."""

import json

import torch

from check_kernel import raw_scores, score_reference
from upstream import flashprefill_native_forward as ops


@torch.inference_mode()
def main():
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    q = torch.randn(1, 1024, 28, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 1024, 4, 128, device="cuda", dtype=torch.bfloat16)
    for name, queries in [("zero_q", torch.zeros_like(q)), ("random_q", q)]:
        torch.compiler.cudagraph_mark_step_begin()
        mean, raw, maximum = raw_scores(queries, k)
        eager = ops.normalize_scores._torchdynamo_orig_callable(raw, maximum)
        compiled = ops.normalize_scores(raw, maximum).clone()
        reference = score_reference(queries, mean)
        future = torch.triu(torch.ones(8, 8, dtype=torch.bool, device="cuda"), diagonal=1)
        result = {
            "case": name,
            "q_stride": queries.stride(), "mean_stride": mean.stride(), "raw_stride": raw.stride(),
            "eager_stride": eager.stride(), "compiled_stride": compiled.stride(),
            "reference_stride": reference.stride(),
            "compiled_vs_eager_max": (compiled - eager).abs().max().item(),
            "eager_vs_reference_max": (eager - reference).abs().max().item(),
            "finite_future_raw_maxima": torch.isfinite(maximum[0, :, :, 0][future]).sum().item(),
            "raw_head0": raw[0, :, :, 0].tolist(),
            "max_head0": maximum[0, :, :, 0].tolist(),
            "eager_head0": eager[0, :, :, 0].tolist(),
            "compiled_head0": compiled[0, :, :, 0].tolist(),
            "reference_head0": reference[0, :, :, 0].tolist(),
        }
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
