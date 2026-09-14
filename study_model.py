"""Qwen2.5-7B full-sequence inference within a 24 GiB GPU; no persistent KV."""

from types import MethodType

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def chunked_forward(module, hidden):
    output = torch.empty_like(hidden)
    for left in range(0, hidden.shape[1], module.token_chunk):
        right = left + module.token_chunk
        output[:, left:right] = module.unchunked_forward(hidden[:, left:right])
    return output


def load_model(name, rope, token_chunk):
    config = AutoConfig.from_pretrained(name)
    config.use_sliding_window = False
    config.sliding_window = None
    if rope == "yarn4":
        # Transformers 4.51.3 derives YaRN's factor from max/original_max.
        config.max_position_embeddings = 131072
        config.rope_scaling = {
            "rope_type": "yarn", "factor": 4.0,
            "original_max_position_embeddings": 32768,
        }
    else:
        config.max_position_embeddings = 32768
        config.rope_scaling = None
    model = AutoModelForCausalLM.from_pretrained(
        name, config=config, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="cuda",
    ).eval()
    modules = [model.model.norm]
    for layer in model.model.layers:
        modules.extend([layer.mlp, layer.input_layernorm, layer.post_attention_layernorm])
    for module in modules:
        module.token_chunk = token_chunk
        module.unchunked_forward = module.forward
        module.forward = MethodType(chunked_forward, module)
    return model


@torch.inference_mode()
def hidden_forward(model, ids):
    torch.compiler.cudagraph_mark_step_begin()
    return model.model(input_ids=ids, use_cache=False).last_hidden_state
