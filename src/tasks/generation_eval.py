"""Utility scoring for generative benchmarks."""
from __future__ import annotations

import torch

from src import metrics as MET
from src import models as M
from src.generate import greedy_continuations


def _mean_metrics(
    predictions: list[str], targets: list[str],
) -> dict[str, float]:
    """Score a batch of continuations.

    Args:
        predictions: Generated strings.
        targets: Reference answers.

    Returns:
        Mean exact match and character-edit similarity.
    """
    return {
        "exact_match": MET.mean_exact_match(predictions, targets),
        "char_edit_similarity": MET.mean_char_edit_similarity(
            predictions, targets,
        ),
    }


def _decode(
    model,
    tokenizer,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    hidden: torch.Tensor,
    k: int,
    max_new_tokens: int,
) -> list[str]:
    """Greedy-decode one activation tensor.

    Args:
        model: Causal language model.
        tokenizer: Tokenizer.
        token_ids: Prompt ids.
        attention_mask: Prompt mask.
        hidden: Prompt cut activations.
        k: Split depth.
        max_new_tokens: Continuation cap.

    Returns:
        One string per row.
    """
    return greedy_continuations(
        model,
        tokenizer,
        token_ids,
        attention_mask,
        hidden,
        k,
        max_new_tokens=max_new_tokens,
        batch_size=M.forward_batch_size(model),
    )


def _average(samples: list[dict[str, float]]) -> dict[str, float]:
    """Average metric dicts key by key.

    Args:
        samples: One dict per noise draw.

    Returns:
        Mean of each key. Empty input returns zeros for both metrics.
    """
    if not samples:
        return {"exact_match": 0.0, "char_edit_similarity": 0.0}
    keys = samples[0].keys()
    return {
        key: float(sum(sample[key] for sample in samples) / len(samples))
        for key in keys
    }


def _noisy_pair(
    model,
    tokenizer,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    clipped_hidden: torch.Tensor,
    targets: list[str],
    k: int,
    max_new_tokens: int,
    cov,
    denoiser_state: dict | None,
    repeats: int,
    seed: int,
    apply_denoiser,
) -> tuple[dict[str, float], dict[str, float]]:
    """Average raw and suppressed scores over noise draws.

    Args:
        model: Causal language model.
        tokenizer: Tokenizer.
        token_ids: Prompt ids.
        attention_mask: Prompt mask.
        clipped_hidden: Clipped prompt activations.
        targets: Reference answers.
        k: Split depth.
        max_new_tokens: Continuation cap.
        cov: Noise covariance.
        denoiser_state: Suppressor state, or ``None``.
        repeats: Number of noise draws.
        seed: Base noise seed.
        apply_denoiser: Callable ``(hidden, state) -> hidden``.

    Returns:
        Mean raw scores and mean suppressed scores.
    """
    raw_scores = []
    private_scores = []
    for draw in range(max(1, repeats)):
        sub_seed = (int(seed) * 1_000_003 + draw) & 0x7FFFFFFFFFFFFFFF
        generator = torch.Generator().manual_seed(sub_seed)
        noise = cov.sample(
            clipped_hidden.shape, generator, device="cpu", dtype=torch.float32,
        )
        noisy = clipped_hidden + noise.to(clipped_hidden.dtype)
        raw_text = _decode(
            model, tokenizer, token_ids, attention_mask,
            noisy, k, max_new_tokens,
        )
        raw_scores.append(_mean_metrics(raw_text, targets))
        if denoiser_state is None:
            private_scores.append(raw_scores[-1])
            continue
        suppressed = apply_denoiser(noisy, denoiser_state)
        private_text = _decode(
            model, tokenizer, token_ids, attention_mask,
            suppressed, k, max_new_tokens,
        )
        private_scores.append(_mean_metrics(private_text, targets))
    return _average(raw_scores), _average(private_scores)


@torch.no_grad()
def score_generation(
    model,
    tokenizer,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    clean_hidden: torch.Tensor,
    clipped_hidden: torch.Tensor,
    targets: list[str],
    k: int,
    max_new_tokens: int,
    cov,
    denoiser_state: dict | None,
    repeats: int,
    seed: int,
    apply_denoiser,
    clean_metrics: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """Score clean, raw, and suppressed continuations.

    Args:
        model: Causal language model.
        tokenizer: Tokenizer.
        token_ids: Prompt ids.
        attention_mask: Prompt mask.
        clean_hidden: Un-noised prompt activations.
        clipped_hidden: Clipped prompt activations.
        targets: Reference answers.
        k: Split depth.
        max_new_tokens: Continuation cap.
        cov: Noise covariance. ``None`` scores the clean channel only.
        denoiser_state: Suppressor state. ``None`` leaves the raw channel.
        repeats: Number of noise draws.
        seed: Base seed for the noise generator.
        apply_denoiser: Callable ``(hidden, state) -> hidden``.
        clean_metrics: Precomputed clean scores. When omitted, the
            clean channel is decoded once.

    Returns:
        ``metric -> {clean, raw, private}``.
    """
    if clean_metrics is None:
        clean_text = _decode(
            model, tokenizer, token_ids, attention_mask,
            clean_hidden, k, max_new_tokens,
        )
        clean = _mean_metrics(clean_text, targets)
    else:
        clean = clean_metrics
    if cov is None:
        return {
            name: {"clean": value, "raw": value, "private": value}
            for name, value in clean.items()
        }
    raw, private = _noisy_pair(
        model, tokenizer, token_ids, attention_mask, clipped_hidden,
        targets, k, max_new_tokens, cov, denoiser_state, repeats, seed,
        apply_denoiser,
    )
    return {
        name: {
            "clean": clean[name],
            "raw": raw[name],
            "private": private[name],
        }
        for name in clean
    }
