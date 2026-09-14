"""Original FlashPrefill V1 and PyTorch Flash SDPA for Transformers 4.51.3."""

import torch
import torch.nn.functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from upstream import flashprefill_native_forward as kernels


class AttentionBackend:
    def __init__(self, alpha=0.08):
        self.method = "dense"
        self.alpha = alpha
        self.record = False
        self.events = []
        self.blocks = []
        self.layer = 0
        self.original_select = kernels.deal_output_score
        kernels.deal_output_score = self.select
        ALL_ATTENTION_FUNCTIONS["sdpa"] = self.forward

    def select(self, *args, **kwargs):
        indices, counts = self.original_select(*args, **kwargs)
        if self.record:
            self.blocks.append((self.layer, counts.detach().clone()))
        return indices, counts

    def forward(self, module, query, key, value, attention_mask,
                scaling, dropout=0.0, **kwargs):
        self.layer = module.layer_idx
        if self.record:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        if self.method == "flashprefill" and query.shape[2] == key.shape[2]:
            q = query.transpose(1, 2).contiguous()
            k = key.transpose(1, 2).contiguous()
            v = value.transpose(1, 2).contiguous()
            output = torch.empty_like(q)
            kernels.flash_prefill(q, k, v, output, 128, 2, 4, self.alpha, 2, 0)
        else:
            # A one-token decode query sees every cached key, including itself.
            output = F.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0,
                is_causal=query.shape[2] > 1, scale=scaling, enable_gqa=True,
            ).transpose(1, 2).contiguous()

        if self.record:
            end.record()
            self.events.append((module.layer_idx, start, end))
        return output, None

    def start_profile(self):
        self.events.clear()
        self.blocks.clear()
        self.record = True

    def finish_profile(self):
        self.record = False
        torch.cuda.synchronize()
        densities = {}
        for layer, counts in self.blocks:
            batch, blocks, heads = counts.shape
            total = batch * heads * blocks * (blocks + 1) // 2
            densities[layer] = counts.sum().item() / total
        rows = [
            {"layer": layer, "attention_ms": start.elapsed_time(end),
             "causal_block_density": densities.get(layer, 1.0)}
            for layer, start, end in self.events
        ]
        self.events.clear()
        self.blocks.clear()
        return rows
