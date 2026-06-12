"""SIPIT attacks: vanilla (Euclidean), Mahalanobis, sequence-MAP/beam.

Strategy is the *minimal* form — for each (prompt, token position t):
  - Build candidate cloud of `top_k` next-token candidates from the model's own
    distribution conditioned on the *true* prefix x_{<t}.
  - For each candidate v, compute h_t(prefix, v) at the cut layer.
  - Score each candidate against the noisy observation a_tilde at position t.
    * vanilla: euclid argmin ||a_tilde - h_t(prefix, v)||
    * mahalanobis: argmin (a_tilde - h)^T Sigma^{-1} (a_tilde - h)
    * sequence-MAP/beam: combine candidate score with model log-prior.
  - Predict the token with highest score; record top-1 token accuracy and
    full-prompt exact-match.

This evaluates per-position attack success on the *cut* tensor (k = split point).
For server-visible non-cut tensors we use a separate similar approach with an
empirically-fit covariance (out of scope for minimal first).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .. import models as M
from .. import split_model as SM
from .. import candidate_sets as CS
from .. import noise as N


@torch.no_grad()
def attack_positions(model, ids: torch.Tensor, mask: torch.Tensor,
                     a_tilde: torch.Tensor, k: int, positions,
                     cov: N.GaussianCov | None = None,
                     prior_weight: float = 0.0,
                     top_k: int = 200, n_rand: int = 50,
                     chunk: int = 256,
                     clouds: dict | None = None,
                     score_fn=None) -> dict:
    """Run SIPIT attack at given positions on the cut tensor a_tilde.

    Args:
      a_tilde: noisy cut activation [B, T, H] on `device`.
      cov: if given, use Mahalanobis distance with Sigma^{-1}; else Euclidean.
      prior_weight: lambda for mixing with model log-prior (sequence-MAP variant).
      clouds: optional dict {position: {"cloud": [B, V', H], "cand": [B, V'],
              "prior_lp": [B, V'] or None}} of pre-built candidate clouds. When
              supplied, the model is NOT called for top-k / cloud / log-prior
              construction at those positions, and `top_k`, `n_rand`, `chunk`
              are ignored. Required for B1's per-(prompt,position) cloud cache;
              `07_eval_attacks.py` continues to call without `clouds=` and gets
              the original on-the-fly behavior.
      score_fn: optional callable (cloud, obs) -> [B, V'] score (higher = better).
              If supplied, it overrides cov-based scoring and is used for
              attackers that need a custom distance (e.g., Wiener posterior-mean
              residual). prior_weight is still applied to the result.

    Returns dict of per-position metrics.
    """
    device = a_tilde.device
    B, T, H = a_tilde.shape
    vocab = model.config.vocab_size
    correct1 = 0; total = 0
    correct1_per_pos = {int(p): {"correct": 0, "total": 0} for p in positions}
    for t in positions:
        if t == 0 or t >= T: continue
        if clouds is not None and int(t) in clouds:
            entry = clouds[int(t)]
            cand = entry["cand"].to(device)
            cloud = entry["cloud"].to(device, a_tilde.dtype)
            cached_prior_lp = entry.get("prior_lp", None)
        else:
            top = CS.topk_candidates(model, ids, position=t, k_top=top_k)
            rand = CS.random_candidates(vocab, B, n_rand, seed=98765 + t).to(device)
            gt = ids[:, t:t+1]
            cand = torch.cat([top, rand, gt], dim=1)            # [B, V'+1]
            cloud = CS.candidate_cloud(model, ids, position=t, candidate_ids=cand,
                                       k_split=k, chunk_size=chunk)   # [B, V', H]
            cloud = cloud.to(device, a_tilde.dtype)
            cached_prior_lp = None
        obs = a_tilde[:, t, :]                                # [B, H]
        diff = cloud - obs.unsqueeze(1)                       # [B, V', H]
        if score_fn is not None:
            score = score_fn(cloud, obs)
        elif cov is not None:
            cov_d = cov.to(device, a_tilde.dtype)
            score = -cov_d.maha(diff)                         # higher is better
        else:
            score = -(diff * diff).sum(dim=-1)
        if prior_weight > 0:
            if cached_prior_lp is not None:
                prior_score = cached_prior_lp.to(device, score.dtype)
            else:
                with torch.no_grad():
                    logits_prior = model(input_ids=ids[:, :t]).logits[:, -1, :]
                    lp = F.log_softmax(logits_prior, dim=-1)      # [B, V]
                prior_score = torch.gather(lp, 1, cand).to(device, score.dtype)
            score = score + prior_weight * prior_score
        argmax = score.argmax(dim=-1)                          # [B]
        pred_token = torch.gather(cand, 1, argmax.unsqueeze(-1)).squeeze(-1)  # [B]
        truth = ids[:, t]
        ok = (pred_token == truth)
        m_t = mask[:, t].bool() if mask is not None else torch.ones_like(truth, dtype=torch.bool)
        correct1 += int((ok & m_t).sum())
        total += int(m_t.sum())
        correct1_per_pos[int(t)]["correct"] = int((ok & m_t).sum())
        correct1_per_pos[int(t)]["total"] = int(m_t.sum())

    return {
        "token_top1": correct1 / max(1, total),
        "n_eval": total,
        "by_position": {int(t): (v["correct"] / max(1, v["total"]) if v["total"] else 0.0)
                        for t, v in correct1_per_pos.items()},
    }
