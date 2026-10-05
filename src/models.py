"""Model loaders and split-point boundaries.

Supported checkpoints:
  - gpt2
  - Qwen/Qwen2.5-0.5B
  - Qwen/Qwen3.5-4B (text decoder of the multimodal checkpoint)

For each architecture we expose load_model, get_blocks, final_norm,
lm_head, n_layers, hidden_size, and vocab_size.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
QWEN25_0_5B = "Qwen/Qwen2.5-0.5B"
QWEN35_4B = "Qwen/Qwen3.5-4B"
SUPPORTED = ["gpt2", QWEN25_0_5B, QWEN35_4B]
DEFAULT_BATCH = 4
DEFAULT_CLOUD_CHUNK = 32


def _cache_writable(path: str) -> bool:
    """Return whether ``path`` or its nearest parent is writable.

    Args:
        path: Cache directory that may not exist yet.

    Returns:
        True when the process can create files there.
    """
    target = Path(path).expanduser()
    try:
        if target.exists():
            return os.access(target, os.W_OK)
        parent = target
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        return os.access(parent, os.W_OK)
    except OSError:
        return False


def _prefer_project_cache() -> None:
    """Point Hugging Face caches at the project when unset or read-only."""
    if (
        not os.environ.get("HF_HOME")
        or not _cache_writable(os.environ["HF_HOME"])
    ):
        os.environ["HF_HOME"] = str(ROOT / ".hf_cache")
    hub = str(Path(os.environ["HF_HOME"]) / "hub")
    if (
        not os.environ.get("HF_HUB_CACHE")
        or not _cache_writable(os.environ["HF_HUB_CACHE"])
    ):
        os.environ["HF_HUB_CACHE"] = hub
    if (
        not os.environ.get("TRANSFORMERS_CACHE")
        or not _cache_writable(os.environ["TRANSFORMERS_CACHE"])
    ):
        os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HUB_CACHE"]


_prefer_project_cache()


def is_qwen35_id(name: str) -> bool:
    """Return whether ``name`` is a Qwen3.5 checkpoint id.

    Args:
        name: Hugging Face id or local directory name.

    Returns:
        True for ids such as ``Qwen/Qwen3.5-4B``.
    """
    token = name.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    token = token.lower().replace("_", ".")
    return token.startswith("qwen3.5")


def default_dtype(name: str, device: str = "cpu") -> torch.dtype:
    """Return the shared weight dtype for every model.

    CUDA devices that support bfloat16 use it. Other CUDA devices
    use float16. CPU stays on float32.

    Args:
        name: Hugging Face model id. Unused.
        device: Torch device string.

    Returns:
        Parameter dtype for ``from_pretrained``.
    """
    del name
    on_cuda = (
        str(device).startswith("cuda") and torch.cuda.is_available()
    )
    if not on_cuda:
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def calibration_batch_size(name: str) -> int:
    """Return the shared activation-cache batch size.

    Args:
        name: Hugging Face model id. Unused.

    Returns:
        Batch size used when a script omits one.
    """
    del name
    return DEFAULT_BATCH


def family(model: nn.Module) -> str:
    """Return the architecture family name.

    Args:
        model: A loaded model.

    Returns:
        ``gpt2``, ``qwen``, or ``qwen35``.
    """
    cls = model.__class__.__name__
    if "GPT2" in cls:
        return "gpt2"
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "language_model"):
        return "qwen35"
    if "Qwen" in cls:
        return "qwen"
    raise ValueError(f"unsupported model class {cls}")


def _qwen_trunk(model: nn.Module) -> nn.Module:
    """Return the Qwen module that owns layers, norm, and embeddings.

    Args:
        model: A Qwen2 or Qwen3.5 model.

    Returns:
        Text decoder module.
    """
    fam = family(model)
    if fam == "qwen35":
        return model.model.language_model
    if fam == "qwen":
        return model.model
    raise ValueError(fam)


def _text_config(model: nn.Module):
    """Return the config that stores text hidden size and vocab size.

    Args:
        model: A loaded model.

    Returns:
        Text config, or the top-level config when it is already text.
    """
    cfg = model.config
    text = getattr(cfg, "text_config", None)
    if text is not None and getattr(text, "hidden_size", None):
        return text
    return cfg


def get_blocks(model: nn.Module) -> nn.ModuleList:
    """Return decoder blocks in order.

    Args:
        model: A loaded model.

    Returns:
        Module list of transformer blocks.
    """
    fam = family(model)
    if fam == "gpt2":
        return model.transformer.h
    if fam in ("qwen", "qwen35"):
        return _qwen_trunk(model).layers
    raise ValueError(fam)


def n_layers(model: nn.Module) -> int:
    """Return the number of decoder blocks.

    Args:
        model: A loaded model.

    Returns:
        Layer count.
    """
    return len(get_blocks(model))


def hidden_size(model: nn.Module) -> int:
    """Return the text-decoder hidden size.

    Args:
        model: A loaded model.

    Returns:
        Hidden width.
    """
    return int(_text_config(model).hidden_size)


def vocab_size(model: nn.Module) -> int:
    """Return the text vocabulary size.

    Args:
        model: A loaded model.

    Returns:
        Vocabulary size used by the LM head.
    """
    return int(_text_config(model).vocab_size)


def final_norm(model: nn.Module) -> nn.Module:
    """Return the norm applied to the final decoder state.

    Args:
        model: A loaded model.

    Returns:
        Final normalization module.
    """
    fam = family(model)
    if fam == "gpt2":
        return model.transformer.ln_f
    if fam in ("qwen", "qwen35"):
        return _qwen_trunk(model).norm
    raise ValueError(fam)


def lm_head(model: nn.Module) -> nn.Module:
    """Return the output embedding / LM head.

    Args:
        model: A loaded model.

    Returns:
        Module mapping hidden states to logits.
    """
    if hasattr(model, "lm_head"):
        return model.lm_head
    return model.get_output_embeddings()


def forward_batch_size(model: nn.Module) -> int:
    """Return the eval batch size for split forwards.

    Args:
        model: A loaded model. Unused.

    Returns:
        Shared batch size for every architecture.
    """
    del model
    return DEFAULT_BATCH


def cloud_chunk_size(model: nn.Module) -> int:
    """Return how many candidate sequences to score at once.

    Args:
        model: A loaded model. Unused.

    Returns:
        Shared chunk size for candidate-cloud forwards.
    """
    del model
    return DEFAULT_CLOUD_CHUNK


def _prepare_tokenizer(name: str):
    """Load a tokenizer and fill a missing pad token.

    Args:
        name: Hugging Face model id.

    Returns:
        Tokenizer with ``pad_token`` set.
    """
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _drop_vision_tower(model: nn.Module) -> None:
    """Remove a vision tower that text-only SST-2 forwards never use.

    Args:
        model: Qwen3.5 conditional-generation model.
    """
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "visual"):
        del inner.visual


def _load_causal_lm(
    name: str,
    dtype: torch.dtype,
    device: str,
) -> nn.Module:
    """Load a plain causal LM and move it to ``device``.

    Args:
        name: Hugging Face model id.
        dtype: Parameter dtype.
        device: Destination device.

    Returns:
        Model still in the mode ``from_pretrained`` left it in.
    """
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    return model.to(device)


def _load_qwen35(
    name: str,
    dtype: torch.dtype,
    device: str,
) -> nn.Module:
    """Load Qwen3.5 text weights and leave the vision tower unloaded.

    The public checkpoint is ``Qwen3_5ForConditionalGeneration``.
    Text SST-2 only needs ``model.language_model`` and ``lm_head``.

    Args:
        name: Hugging Face model id, normally ``Qwen/Qwen3.5-4B``.
        dtype: Parameter dtype.
        device: Destination device.

    Returns:
        Conditional-generation model with the vision tower removed.
    """
    try:
        from transformers import Qwen3_5ForConditionalGeneration
    except ImportError as exc:
        raise ImportError(
            "Qwen3.5 requires transformers>=5.5"
        ) from exc
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        name,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    _drop_vision_tower(model)
    return model.to(device)


def load_model(
    name: str,
    dtype: torch.dtype | None = None,
    device: str = "cuda:0",
):
    """Load a supported model and tokenizer in eval mode.

    Args:
        name: Hugging Face model id. ``Qwen/Qwen3.5-4B`` selects the
            Qwen3.5 text decoder.
        dtype: Parameter dtype. ``None`` uses ``default_dtype``.
        device: Destination device.

    Returns:
        ``(model, tokenizer)``.
    """
    if dtype is None:
        dtype = default_dtype(name, device)
    tok = _prepare_tokenizer(name)
    if is_qwen35_id(name):
        model = _load_qwen35(name, dtype, device)
    else:
        model = _load_causal_lm(name, dtype, device)
    model.eval()
    return model, tok


def embed(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Return the activation that block 0 expects.

    GPT-2 adds token and position embeddings. Qwen2 and Qwen3.5
    return token embeddings only; rotary is applied inside attention.

    Args:
        model: A loaded model.
        input_ids: Token ids of shape ``[batch, time]``.

    Returns:
        Embedding tensor of shape ``[batch, time, hidden]``.
    """
    fam = family(model)
    if fam == "gpt2":
        trunk = model.transformer
        token_emb = trunk.wte(input_ids)
        seq = input_ids.shape[1]
        pos = torch.arange(seq, device=input_ids.device).unsqueeze(0)
        return token_emb + trunk.wpe(pos)
    if fam in ("qwen", "qwen35"):
        return _qwen_trunk(model).embed_tokens(input_ids)
    raise ValueError(fam)


def position_ids(seq_len: int, device: torch.device | str) -> torch.Tensor:
    """Return positions ``0 .. seq_len-1``.

    Args:
        seq_len: Sequence length.
        device: Torch device.

    Returns:
        Long tensor of shape ``[1, seq_len]``.
    """
    return torch.arange(seq_len, device=device).unsqueeze(0)
