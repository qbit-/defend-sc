"""Model loaders and split-point boundaries.

For each supported architecture we expose:
  - load_model(name, dtype, device)
  - get_blocks(model)              -> list of transformer blocks (modifiable in-place)
  - input_layernorm_module(model)  -> module applied to embeddings before block 0 (or None)
  - final_norm(model)
  - lm_head(model)
  - n_layers(model)
  - hidden_size(model)
"""
from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SUPPORTED = ["gpt2", "Qwen/Qwen2.5-0.5B"]


def load_model(name: str, dtype=torch.float32, device="cuda:0"):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(device)
    model.eval()
    return model, tok


def family(model) -> str:
    cls = model.__class__.__name__
    if "GPT2" in cls:
        return "gpt2"
    if "Qwen" in cls:
        return "qwen"
    raise ValueError(f"unsupported model class {cls}")


def get_blocks(model):
    fam = family(model)
    if fam == "gpt2":
        return model.transformer.h
    if fam == "qwen":
        return model.model.layers
    raise ValueError(fam)


def n_layers(model) -> int:
    return len(get_blocks(model))


def hidden_size(model) -> int:
    return model.config.hidden_size


def final_norm(model):
    fam = family(model)
    if fam == "gpt2":
        return model.transformer.ln_f
    if fam == "qwen":
        return model.model.norm
    raise ValueError(fam)


def lm_head(model):
    return model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()


def embed(model, input_ids):
    """Return embedding-side activation that block 0 expects.

    For GPT-2: token_emb + position_emb + dropout(pdrop=0 in eval).
    For Qwen2: token embeddings only; rotary is handled inside the attention layer
    via position_ids / rope cache, not added to the input.
    """
    fam = family(model)
    if fam == "gpt2":
        t = model.transformer
        wte = t.wte(input_ids)
        seq = input_ids.shape[1]
        pos_ids = torch.arange(seq, device=input_ids.device).unsqueeze(0)
        wpe = t.wpe(pos_ids)
        return wte + wpe
    if fam == "qwen":
        return model.model.embed_tokens(input_ids)
    raise ValueError(fam)


def position_ids(seq_len: int, device) -> torch.Tensor:
    return torch.arange(seq_len, device=device).unsqueeze(0)
