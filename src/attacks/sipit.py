"""SIPIT token recovery at the cut activation.

Vanilla uses Euclidean distance, Mahalanobis uses the noise covariance,
sequence-MAP adds the model log-prior, and ``score_fn`` covers the exact
attacker. With ``independent_recovery``, every position uses the true
prefix. Otherwise each call writes its guess into a private copy of the
prompt, and later positions condition on that copy.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F

from .. import candidate_sets as CS
from .. import models as M
from .. import noise as N


def _active_positions(
    positions: Iterable[int], length: int,
) -> list[int]:
    """Return attacked indices inside the sequence, low to high.

    Args:
        positions: Requested token indices.
        length: Sequence length.

    Returns:
        Sorted positions ``t`` with ``0 < t < length``.
    """
    chosen = {
        int(position)
        for position in positions
        if 0 < int(position) < length
    }
    return sorted(chosen)


def _scores(
    cloud: torch.Tensor,
    obs: torch.Tensor,
    cov: N.GaussianCov | None,
    score_fn: Callable[..., torch.Tensor] | None,
) -> torch.Tensor:
    """Return higher-is-better scores for one position.

    Args:
        cloud: Candidate cut states, ``[B, V', H]``.
        obs: Observed cut state, ``[B, H]``.
        cov: Mahalanobis covariance, or ``None`` for Euclidean.
        score_fn: Optional ``(cloud, obs) -> [B, V']`` scorer.

    Returns:
        Scores, ``[B, V']``.
    """
    if score_fn is not None:
        return score_fn(cloud, obs)
    diff = cloud - obs.unsqueeze(1)
    if cov is None:
        return -(diff * diff).sum(dim=-1)
    cov_d = cov.to(cloud.device, cloud.dtype)
    return -cov_d.maha(diff)


def _with_prior(
    score: torch.Tensor,
    cand: torch.Tensor,
    prior_lp: torch.Tensor | None,
    model: Any,
    prefix_ids: torch.Tensor,
    position: int,
    prior_weight: float,
) -> torch.Tensor:
    """Add the token log-prior when ``prior_weight`` is positive.

    Args:
        score: Candidate scores, ``[B, V']``.
        cand: Candidate ids, ``[B, V']``.
        prior_lp: Cached log-priors, or ``None`` to compute them.
        model: Causal language model.
        prefix_ids: Tokens that form the attacker prefix.
        position: Index being scored.
        prior_weight: Mixture weight. Zero leaves ``score`` unchanged.

    Returns:
        Updated scores, ``[B, V']``.
    """
    if prior_weight <= 0:
        return score
    if prior_lp is None:
        logits = model(input_ids=prefix_ids[:, :position]).logits[:, -1, :]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        prior_lp = torch.gather(log_probs, 1, cand)
    mix = prior_lp.to(device=score.device, dtype=score.dtype)
    return score + prior_weight * mix


def _write_prediction(
    recovered: torch.Tensor,
    position: int,
    predicted: torch.Tensor,
    mask: torch.Tensor | None,
) -> None:
    """Store guessed tokens on the private rows of ``position``.

    Args:
        recovered: Attacker prompt, edited in place, ``[B, T]``.
        position: Token index just scored.
        predicted: Chosen token id per row, ``[B]``.
        mask: Privacy mask, ``[B, T]``. ``None`` updates every row.
    """
    if mask is None:
        recovered[:, position] = predicted
        return
    chosen = mask[:, position].bool()
    recovered[chosen, position] = predicted[chosen]


def _tally(
    predicted: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor | None,
    position: int,
) -> tuple[int, int]:
    """Count correct guesses at one position.

    Args:
        predicted: Chosen ids, ``[B]``.
        truth: Real ids at this position, ``[B]``.
        mask: Privacy or attention mask, ``[B, T]``, or ``None``.
        position: Column of ``mask`` to apply.

    Returns:
        Correct count and number of scored rows.
    """
    ok = predicted == truth
    if mask is None:
        active = torch.ones_like(ok)
    else:
        active = mask[:, position].bool()
    hits = int((ok & active).sum())
    count = int(active.sum())
    return hits, count


def _attack_report(
    correct: int,
    total: int,
    per_pos: dict[int, dict[str, int]],
) -> dict:
    """Pack token accuracy into the eval result.

    Args:
        correct: Total correct token guesses.
        total: Total scored token positions.
        per_pos: Per-position correct and total counts.

    Returns:
        ``token_top1``, ``n_eval``, and ``by_position``.
    """
    rates = {
        position: (
            counts["correct"] / counts["total"]
            if counts["total"] else 0.0
        )
        for position, counts in per_pos.items()
    }
    return {
        "token_top1": correct / max(1, total),
        "n_eval": total,
        "by_position": rates,
    }


def _cached_bundle(
    entry: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Load one precomputed true-prefix cloud.

    Args:
        entry: Cache record with ``cand``, ``cloud``, and ``prior_lp``.
        device: Destination device.
        dtype: Cut-activation dtype.

    Returns:
        Candidate ids, cut states, and log-priors.
    """
    cand = entry["cand"].to(device)
    cloud = entry["cloud"].to(device=device, dtype=dtype)
    prior = entry.get("prior_lp", None)
    return cand, cloud, prior


def _legacy_bundle(
    model: Any,
    ids: torch.Tensor,
    position: int,
    k: int,
    top_k: int,
    n_rand: int,
    chunk: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, None]:
    """Build the on-the-fly cloud used when no cache is supplied.

    Args:
        model: Causal language model.
        ids: True prompt tokens, ``[B, T]``.
        position: Index being guessed.
        k: Split depth.
        top_k: Model candidates to keep.
        n_rand: Random vocab candidates to add.
        chunk: Cloud forward chunk size.
        device: Destination device.
        dtype: Cut-activation dtype.

    Returns:
        Candidate ids, cut states, and ``None`` for the log-prior.
    """
    vocab = M.vocab_size(model)
    batch = ids.shape[0]
    top = CS.topk_candidates(model, ids, position=position, k_top=top_k)
    rand = CS.random_candidates(
        vocab, batch, n_rand, seed=98765 + position,
    ).to(device)
    truth = ids[:, position:position + 1]
    cand = torch.cat([top, rand, truth], dim=1)
    cloud = CS.candidate_cloud(
        model, ids, position=position, candidate_ids=cand,
        k_split=k, chunk_size=chunk,
    )
    return cand.to(device), cloud.to(device=device, dtype=dtype), None


def _recovered_bundle(
    model: Any,
    prefix_ids: torch.Tensor,
    truth_ids: torch.Tensor,
    position: int,
    k: int,
    top_k: int,
    n_rand: int,
    chunk: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a candidate cloud from the attacker prefix.

    The true token is still offered as a candidate. Top-k ids and the
    log-prior both come from ``prefix_ids``, not from the original prompt.

    Args:
        model: Causal language model.
        prefix_ids: Tokens conditioned on, ``[B, T]``.
        truth_ids: Original prompt tokens, ``[B, T]``.
        position: Index being guessed.
        k: Split depth.
        top_k: Model candidates to keep.
        n_rand: Random vocab candidates to add.
        chunk: Cloud forward chunk size.
        device: Destination device.
        dtype: Cut-activation dtype.

    Returns:
        Candidate ids, cut states, and candidate log-priors.
    """
    vocab = M.vocab_size(model)
    batch = prefix_ids.shape[0]
    logits = model(input_ids=prefix_ids[:, :position]).logits[:, -1, :]
    top = logits.topk(top_k, dim=-1).indices
    rand = CS.random_candidates(
        vocab, batch, n_rand, seed=98765 + position,
    ).to(device)
    truth = truth_ids[:, position:position + 1]
    cand = torch.cat([top, rand, truth], dim=1)
    cloud = CS.candidate_cloud(
        model, prefix_ids, position=position, candidate_ids=cand,
        k_split=k, chunk_size=chunk,
    )
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    prior = torch.gather(log_probs, 1, cand)
    cast = cloud.to(device=device, dtype=dtype)
    return cand.to(device), cast, prior


def _bundle_at(
    model: Any,
    truth_ids: torch.Tensor,
    prefix_ids: torch.Tensor,
    position: int,
    k: int,
    top_k: int,
    n_rand: int,
    chunk: int,
    clouds: dict | None,
    independent_recovery: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Choose the candidate cloud for one attacked position.

    Args:
        model: Causal language model.
        truth_ids: Original prompt tokens.
        prefix_ids: Tokens the attacker conditions on.
        position: Index being guessed.
        k: Split depth.
        top_k: Model candidates to keep.
        n_rand: Random vocab candidates to add.
        chunk: Cloud forward chunk size.
        clouds: True-prefix cache. Used only for independent recovery.
        independent_recovery: Keep the true prefix when true.
        device: Destination device.
        dtype: Cut-activation dtype.

    Returns:
        Candidate ids, cut states, and optional log-priors.
    """
    cached = (
        independent_recovery
        and clouds is not None
        and int(position) in clouds
    )
    if cached:
        return _cached_bundle(clouds[int(position)], device, dtype)
    if independent_recovery:
        return _legacy_bundle(
            model, truth_ids, position, k, top_k, n_rand, chunk,
            device, dtype,
        )
    return _recovered_bundle(
        model, prefix_ids, truth_ids, position, k, top_k, n_rand,
        chunk, device, dtype,
    )


def _predict_at(
    model: Any,
    ids: torch.Tensor,
    prefix: torch.Tensor,
    mask: torch.Tensor | None,
    observed: torch.Tensor,
    position: int,
    k: int,
    top_k: int,
    n_rand: int,
    chunk: int,
    clouds: dict | None,
    cov: N.GaussianCov | None,
    prior_weight: float,
    score_fn: Callable[..., torch.Tensor] | None,
    independent_recovery: bool,
    recovered: torch.Tensor | None,
) -> tuple[int, int]:
    """Score one position and store the guess when recovering.

    Args:
        model: Causal language model.
        ids: True prompt tokens.
        prefix: Tokens the attacker conditions on.
        mask: Privacy mask, or ``None``.
        observed: Cut activations, ``[B, T, H]``.
        position: Index being guessed.
        k: Split depth.
        top_k: Model candidates to keep.
        n_rand: Random vocab candidates to add.
        chunk: Cloud forward chunk size.
        clouds: True-prefix cache, or ``None``.
        cov: Mahalanobis covariance, or ``None``.
        prior_weight: Log-prior mixture weight.
        score_fn: Optional custom scorer.
        independent_recovery: Use the true prefix when true.
        recovered: Prompt edited in place, or ``None``.

    Returns:
        Correct count and number of scored rows.
    """
    device = observed.device
    dtype = observed.dtype
    cand, cloud, prior_lp = _bundle_at(
        model, ids, prefix, position, k, top_k, n_rand, chunk,
        clouds, independent_recovery, device, dtype,
    )
    obs = observed[:, position, :]
    score = _scores(cloud, obs, cov, score_fn)
    score = _with_prior(
        score, cand, prior_lp, model, prefix, position, prior_weight,
    )
    choice = score.argmax(dim=-1)
    predicted = torch.gather(
        cand, 1, choice.unsqueeze(-1),
    ).squeeze(-1)
    if recovered is not None:
        _write_prediction(recovered, position, predicted, mask)
    return _tally(predicted, ids[:, position], mask, position)


@torch.no_grad()
def attack_positions(
    model: Any,
    ids: torch.Tensor,
    mask: torch.Tensor | None,
    a_tilde: torch.Tensor,
    k: int,
    positions: Iterable[int],
    cov: N.GaussianCov | None = None,
    prior_weight: float = 0.0,
    top_k: int = 200,
    n_rand: int = 50,
    chunk: int = 256,
    clouds: dict | None = None,
    score_fn: Callable[..., torch.Tensor] | None = None,
    independent_recovery: bool = False,
) -> dict:
    """Run SIPIT at ``positions`` on the cut tensor.

    Positions are scored from left to right. Unless
    ``independent_recovery`` is set, the guessed token replaces the
    original token on rows where ``mask`` is 1. The next position's
    candidates, cut states, and log-prior all use that recovered prompt.
    Rows with mask 0 keep the original token. Tokens that are not in
    ``positions`` stay as in the original prompt.

    Args:
        model: Causal language model.
        ids: True prompt tokens, ``[B, T]``.
        mask: Privacy mask, ``[B, T]``. ``None`` scores every row.
        a_tilde: Cut activations, ``[B, T, H]``.
        k: Split depth.
        positions: Token indices to recover.
        cov: Mahalanobis covariance, or ``None`` for Euclidean.
        prior_weight: Log-prior mixture weight.
        top_k: Model candidates to keep when no cache is used.
        n_rand: Random vocab candidates to add.
        chunk: Cloud forward chunk size.
        clouds: True-prefix clouds. Ignored unless recovery is
            independent.
        score_fn: Optional custom candidate scorer.
        independent_recovery: Use the true prefix at every position.

    Returns:
        Top-1 token accuracy, the number of scored tokens, and
        per-position rates.
    """
    length = int(a_tilde.shape[1])
    recovered = None if independent_recovery else ids.clone()
    ordered = _active_positions(positions, length)
    per_pos = {
        position: {"correct": 0, "total": 0} for position in ordered
    }
    correct = 0
    total = 0
    for position in ordered:
        prefix = ids if recovered is None else recovered
        hits, count = _predict_at(
            model, ids, prefix, mask, a_tilde, position, k, top_k,
            n_rand, chunk, clouds, cov, prior_weight, score_fn,
            independent_recovery, recovered,
        )
        correct += hits
        total += count
        per_pos[position]["correct"] = hits
        per_pos[position]["total"] = count
    return _attack_report(correct, total, per_pos)
