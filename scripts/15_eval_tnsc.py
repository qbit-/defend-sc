"""Setting G TNSC evaluation scaffold for SST-2/logit pilot artifacts."""
from __future__ import annotations
import os, sys, argparse, json, csv
from pathlib import Path
import math
from typing import Any, Callable, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))

from src import candidate_sets as CS
from src import metrics as MET
from src import models as M
from src import noise as N
from src import split_model as SM
from src import util
from src import seeding
from src.attacks import exact_dist as ED
from src.attacks import sipit as SIP
from src.tasks import get_task
from src.tasks.generation_eval import score_generation


VARIANT_TO_COV = {
    "clean_no_noise": None,
    "b1_lowrank_struct": "lowrank_struct",
    "b1_lowrank_struct_private_suppressor": "lowrank_struct",
}
DEFAULT_VARIANTS = tuple(VARIANT_TO_COV)
DEFAULT_REPEATS = (1, 4, 16)
REPEAT_POLICIES = ("independent_average", "sticky_same_noise", "mixed_retry")
A_TERMS = ("task_logit_surrogate", "task_subspace", "ood_diag")


def model_safe(model_id: str) -> str:
    return util.model_slug(model_id)


def cache_path(model_id: str, k: int, task: str = "sst2") -> Path:
    return util.art_path(
        "activations", model_id, task, f"split_{k}", "cache.pt",
    )


def task_weights_path(model_id: str, k: int, task: str = "sst2") -> Path:
    return util.art_path(
        "activations", model_id, task, f"split_{k}", "task_weights.pt",
    )


def task_bias_path(model_id: str, k: int, task: str = "sst2") -> Path:
    return util.art_path(
        "activations", model_id, task, f"split_{k}", "task_bias.pt",
    )


def metadata_path(model_id: str, k: int, task: str = "sst2") -> Path:
    return util.art_path(
        "activations", model_id, task, f"split_{k}", "metadata.json",
    )


def cov_path(model_id: str, k: int, sigma0_frac: float, rank: int, family: str,
             task: str = "sst2") -> Path:
    return util.art_path(
        "covariances", model_id, task, f"split_{k}",
        f"sigma0_{sigma0_frac:g}_r{rank}__{family}.pt",
    )


def private_denoiser_path(model_id: str, k: int, sigma0_frac: float, rank: int,
                          suffix: str = "tnsc_gaussian", task: str = "sst2") -> Path:
    return util.art_path(
        "private_denoisers", model_id, task, f"split_{k}",
        f"sigma0_{sigma0_frac:g}_r{rank}__{suffix}", "private_denoiser.pt",
    )


def cloud_cache_dir(model_id: str, task: str, k: int) -> Path:
    return util.art_path("sipit_clouds", model_id, task, f"split_{k}")


def eval_dir(model_id: str, task: str) -> Path:
    """Return the eval directory for one model and benchmark.

    Args:
        model_id: Hugging Face model id.
        task: Benchmark name.

    Returns:
        ``artifacts/evals/<model>/<task>``.
    """
    return util.art_path("evals", model_id, task)


def load_cov(model_id: str, k: int, sigma0_frac: float, rank: int, family: str,
             task: str = "sst2"):
    p = cov_path(model_id, k, sigma0_frac, rank, family, task=task)
    if not p.exists():
        return None, p
    d = torch.load(p, weights_only=True)
    cov = N.GaussianCov(name=d["name"], hidden=d["hidden"], sigma0=d["sigma0"],
                        U=d.get("U"), lam=d.get("lam"), note=d.get("note", ""))
    return cov, p


def load_private_denoiser_state(model_id: str, k: int, sigma0_frac: float, rank: int,
                                suffix: str = "tnsc_gaussian", task: str = "sst2"):
    p = private_denoiser_path(model_id, k, sigma0_frac, rank, suffix=suffix, task=task)
    if not p.exists():
        return None, p
    return torch.load(p, weights_only=False), p


def apply_private_denoiser(y: torch.Tensor, state: dict) -> torch.Tensor:
    """Apply the lowrank_struct server-private suppressor (shrink span(U_eta))."""
    if state.get("denoiser_type") != "private_lowrank_struct_suppressor":
        raise ValueError(f"unsupported private denoiser type {state.get('denoiser_type')!r}")
    hidden = int(state["hidden"])
    if hidden != y.shape[-1]:
        raise ValueError(f"private suppressor hidden={hidden} does not match y hidden={y.shape[-1]}")
    mean = state.get("mean", state.get("mu")).to(y.device, y.dtype)
    U_eta = state["U_eta"].to(y.device, y.dtype)
    gamma = state["gamma"].to(y.device, y.dtype)
    centered = y - mean
    coeff = centered @ U_eta
    return y - (coeff * gamma) @ U_eta.T


def cov_div_m(cov: N.GaussianCov, m: int) -> N.GaussianCov:
    if m <= 1:
        return cov
    return N.GaussianCov(
        name=cov.name,
        hidden=cov.hidden,
        sigma0=cov.sigma0 / math.sqrt(m),
        U=cov.U,
        lam=(cov.lam / m) if cov.lam is not None else None,
        note=f"{cov.note}/m={m}",
    )


def effective_repeat_count(policy: str, m: int) -> int:
    if policy == "independent_average":
        return m
    if policy == "sticky_same_noise":
        return 1
    if policy == "mixed_retry":
        return max(1, (m + 1) // 2)
    raise ValueError(f"unknown repeat policy {policy!r}")


def materialize_realizations(clipped_a: torch.Tensor, cov: N.GaussianCov, K: int,
                             base_seed: int, dtype) -> list[torch.Tensor]:
    out = []
    for r_idx in range(K):
        sub_seed = (int(base_seed) * 1_000_003 + r_idx) & 0x7FFFFFFFFFFFFFFF
        gen = torch.Generator().manual_seed(sub_seed)
        eta = cov.sample(clipped_a.shape, gen, device="cpu", dtype=torch.float32)
        out.append((clipped_a + eta.to(clipped_a.dtype)).to(dtype).clone())
    return out


def observation_for_policy(realizations: list[torch.Tensor], policy: str, m: int) -> torch.Tensor:
    n_unique = effective_repeat_count(policy, m)
    return torch.stack(realizations[:n_unique], dim=0).mean(dim=0)


def effective_cov_for_policy(cov: N.GaussianCov, policy: str, m: int) -> N.GaussianCov:
    return cov_div_m(cov, effective_repeat_count(policy, m))


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    phat = successes / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return max(0.0, center - half), min(1.0, center + half)


def accuracy_counts(rate: float, total: int) -> tuple[int, int]:
    if total <= 0 or not isinstance(rate, (int, float)):
        return 0, 0
    return int(round(float(rate) * total)), total


@torch.no_grad()
def build_or_load_clouds(model, model_id: str, k: int, ids: torch.Tensor,
                         positions: Iterable[int], top_k: int, n_rand: int,
                         chunk: int, device, task: str = "sst2") -> dict:
    out_dir = cloud_cache_dir(model_id, task, k)
    out_dir.mkdir(parents=True, exist_ok=True)
    import hashlib
    prompt_hash = hashlib.sha256(ids.cpu().contiguous().numpy().tobytes()).hexdigest()[:16]
    pos_key = "-".join(str(int(p)) for p in positions)
    cache_file = out_dir / f"tnsc_clouds_v2_n{ids.shape[0]}_pos{pos_key}_top{top_k}_rand{n_rand}_{prompt_hash}.pt"
    if cache_file.exists():
        blob = torch.load(cache_file, weights_only=False)
        if all(int(t) in blob for t in positions if 0 < int(t) < ids.shape[1]):
            return blob

    ids_dev = ids.to(device)
    vocab = M.vocab_size(model)
    B, T = ids.shape
    blob = {}
    for t in positions:
        t = int(t)
        if t == 0 or t >= T:
            continue
        prior_logits = model(input_ids=ids_dev[:, :t]).logits[:, -1, :]
        top = prior_logits.topk(top_k, dim=-1).indices
        rand = CS.random_candidates(vocab, B, n_rand, seed=98765 + t).to(device)
        truth = ids_dev[:, t:t + 1]
        cand = torch.cat([top, rand, truth], dim=1)
        cloud = CS.candidate_cloud(model, ids_dev, position=t, candidate_ids=cand,
                                   k_split=k, chunk_size=chunk)
        prior_lp = torch.log_softmax(prior_logits.float(), dim=-1)
        blob[t] = {
            "cand": cand.cpu(),
            "cloud": cloud.cpu(),
            "prior_lp": torch.gather(prior_lp, 1, cand).cpu(),
        }
    torch.save(blob, cache_file)
    return blob


@torch.no_grad()
def _server_logits_chunk(
    model: torch.nn.Module,
    k: int,
    ids: torch.Tensor,
    mask: torch.Tensor,
    ans_pos: torch.Tensor,
    hidden: torch.Tensor,
    task_weights: torch.Tensor,
    device: str,
    dtype: torch.dtype,
    task_bias: torch.Tensor | None = None,
    label_token_ids: list | None = None,
) -> torch.Tensor:
    """Score one batch of split activations.

    Args:
        model: Loaded causal model.
        k: Split depth.
        ids: Token ids for this batch.
        mask: Attention mask for this batch.
        ans_pos: Answer positions for this batch.
        hidden: Cut activations for this batch.
        task_weights: Label rows of the LM head.
        device: Torch device.
        dtype: Activation dtype.
        task_bias: Optional label bias.
        label_token_ids: Verbalizer token ids, when known.

    Returns:
        Label logits of shape ``[batch, n_classes]`` on CPU.
    """
    out = SM.split_run(
        model, ids.to(device), k=k,
        hidden_override=hidden.to(device, dtype),
        attention_mask=mask.to(device),
    )
    if label_token_ids is not None:
        pos = ans_pos.to(device).view(-1, 1, 1)
        pos = pos.expand(-1, 1, out["logits"].shape[-1])
        last = out["logits"].gather(1, pos).squeeze(1)
        label_ids = torch.tensor(label_token_ids, device=device)
        return last.index_select(-1, label_ids).cpu().float()
    from src import sst2_data as sst2

    feat = sst2.gather_answer_position(
        out["server_out"], ans_pos.to(device),
    ).cpu().float()
    logits = feat @ task_weights.float().T
    if task_bias is not None:
        logits = logits + task_bias.float()
    return logits


@torch.no_grad()
def server_logits_for(
    model: torch.nn.Module,
    k: int,
    ids: torch.Tensor,
    mask: torch.Tensor,
    ans_pos: torch.Tensor,
    hidden: torch.Tensor,
    task_weights: torch.Tensor,
    device: str,
    dtype: torch.dtype,
    task_bias: torch.Tensor | None = None,
    label_token_ids: list | None = None,
) -> torch.Tensor:
    """Score split activations, chunked for Qwen3.5.

    Args:
        model: Loaded causal model.
        k: Split depth.
        ids: Token ids.
        mask: Attention mask.
        ans_pos: Answer positions.
        hidden: Cut activations.
        task_weights: Label rows of the LM head.
        device: Torch device.
        dtype: Activation dtype.
        task_bias: Optional label bias.
        label_token_ids: Verbalizer token ids, when known.

    Returns:
        Label logits of shape ``[batch, n_classes]`` on CPU.
    """
    batch = M.forward_batch_size(model)
    n = ids.shape[0]
    if batch <= 0 or n <= batch:
        return _server_logits_chunk(
            model, k, ids, mask, ans_pos, hidden, task_weights,
            device, dtype, task_bias, label_token_ids,
        )
    parts = []
    for start in range(0, n, batch):
        stop = min(start + batch, n)
        parts.append(_server_logits_chunk(
            model, k, ids[start:stop], mask[start:stop],
            ans_pos[start:stop], hidden[start:stop], task_weights,
            device, dtype, task_bias, label_token_ids,
        ))
    return torch.cat(parts, dim=0)


def _class_scores(
    clean: float, raw: float, private: float,
) -> dict[str, dict[str, float]]:
    """Return SST-2 accuracy in the shared metric layout.

    Args:
        clean: Clean-channel accuracy.
        raw: Noisy-channel accuracy.
        private: Suppressed-channel accuracy.

    Returns:
        ``accuracy -> {clean, raw, private}``.
    """
    return {
        "accuracy": {"clean": clean, "raw": raw, "private": private},
    }


@torch.no_grad()
def utility_metrics(model, k: int, cache: dict, task_weights, task_bias, label_token_ids,
                    cov: N.GaussianCov | None, private_denoiser_state: dict | None,
                    K: int, seed: int, device, dtype) -> dict:
    test = cache["test"]
    ids = test["ids"]
    mask = test["mask"]
    ans_pos = test["ans_pos"]
    labels = test["labels"]
    clipped_a = test["clipped_a"]
    clean_logits = test["label_logits_clean"].float()
    acc_clean = MET.accuracy(clean_logits, labels)

    if cov is None:
        return _class_scores(acc_clean, acc_clean, acc_clean), {
            "kl_no_corr": 0.0,
            "kl_with_corr": 0.0,
            "top1_no_corr": 1.0,
            "top1_with_corr": 1.0,
            "clean_distortion": 0.0,
            "tame_C_sum": 0.0,
            "tame_V_sum": 0.0,
            "margin_cert_noisy": 1.0,
            "ood_maha_mean": 0.0,
            "ood_maha_p95": 0.0,
            "bootstrap_ci_low": "",
            "bootstrap_ci_high": "",
        }

    logits_no, logits_corr = [], []
    clean_dist = []
    noisy_ood = []
    clean_mean = clipped_a.float().reshape(-1, clipped_a.shape[-1]).mean(dim=0)
    clean_var = clipped_a.float().reshape(-1, clipped_a.shape[-1]).var(dim=0, unbiased=False)
    for i in range(K):
        gen = torch.Generator().manual_seed((seed * 1_000_003 + i) & 0x7FFFFFFFFFFFFFFF)
        eta = cov.sample(clipped_a.shape, gen, device="cpu", dtype=torch.float32)
        noisy = clipped_a + eta.to(clipped_a.dtype)
        if private_denoiser_state is not None:
            hidden_corr = apply_private_denoiser(noisy, private_denoiser_state)
        else:
            hidden_corr = noisy
        logits_i = server_logits_for(
            model, k, ids, mask, ans_pos, noisy, task_weights, device, dtype,
            task_bias=task_bias, label_token_ids=label_token_ids)
        logits_c = server_logits_for(model, k, ids, mask, ans_pos, hidden_corr,
                                     task_weights, device, dtype, task_bias=task_bias,
                                     label_token_ids=label_token_ids)
        logits_no.append(logits_i)
        logits_corr.append(logits_c)
        if private_denoiser_state is not None:
            clean_d = apply_private_denoiser(clipped_a, private_denoiser_state)
        else:
            clean_d = clipped_a
        clean_dist.append(float((clean_d.float() - clipped_a.float()).norm(dim=-1).mean()
                                / clipped_a.float().norm(dim=-1).mean().clamp(min=1e-12)))
        noisy_ood.append(MET.diag_maha_ood(noisy, clean_mean, clean_var).reshape(-1))

    mean_no = torch.stack(logits_no, dim=0).mean(dim=0)
    mean_corr = torch.stack(logits_corr, dim=0).mean(dim=0)
    logits_no_bkc = torch.stack(logits_no, dim=1)
    logits_corr_bkc = torch.stack(logits_corr, dim=1)
    tame_no = MET.tame_bias_variance(clean_logits, logits_no_bkc)
    corr_ok = (logits_corr_bkc.argmax(dim=-1) == labels[:, None]).float().reshape(-1)
    boot_low, boot_high = MET.bootstrap_ci(corr_ok, n_boot=200, seed=seed)
    ood = torch.cat(noisy_ood) if noisy_ood else torch.tensor([0.0])
    raw_acc = MET.accuracy(mean_no, labels)
    private_acc = MET.accuracy(mean_corr, labels)
    return _class_scores(acc_clean, raw_acc, private_acc), {
        "kl_no_corr": MET.kl_logits(clean_logits, mean_no),
        "kl_with_corr": MET.kl_logits(clean_logits, mean_corr),
        "top1_no_corr": MET.top1_agreement(clean_logits, mean_no),
        "top1_with_corr": MET.top1_agreement(clean_logits, mean_corr),
        "clean_distortion": sum(clean_dist) / max(1, len(clean_dist)),
        "tame_C_sum": tame_no["C_sum"],
        "tame_V_sum": tame_no["V_sum"],
        "margin_cert_noisy": MET.margin_cert_rate(clean_logits, logits_no_bkc, labels),
        "ood_maha_mean": float(ood.float().mean()),
        "ood_maha_p95": float(torch.quantile(ood.float(), 0.95)),
        "bootstrap_ci_low": boot_low,
        "bootstrap_ci_high": boot_high,
    }


def _attack_clouds(
    model: Any,
    model_id: str,
    k: int,
    ids: torch.Tensor,
    positions: tuple[int, ...],
    device: str,
    task: str,
    independent_recovery: bool,
) -> dict | None:
    """Return true-prefix clouds, or none for sequential recovery.

    Args:
        model: Causal language model.
        model_id: Checkpoint id.
        k: Split depth.
        ids: Prompt tokens.
        positions: Attack indices.
        device: Torch device.
        task: Benchmark name.
        independent_recovery: Build clouds only when this is true.

    Returns:
        Cloud cache, or ``None`` when guesses are fed back.
    """
    if not independent_recovery:
        return None
    return build_or_load_clouds(
        model, model_id, k, ids, positions, top_k=80, n_rand=20,
        chunk=M.cloud_chunk_size(model), device=device, task=task,
    )


def _run_attack(
    model: Any,
    ids: torch.Tensor,
    mask: torch.Tensor,
    observed: torch.Tensor,
    k: int,
    positions: tuple[int, ...],
    clouds: dict | None,
    cov: N.GaussianCov | None,
    prior_weight: float,
    score_fn: Callable[..., torch.Tensor] | None,
    independent_recovery: bool,
) -> dict:
    """Run one SIPIT variant on its own recovered prompt.

    Args:
        model: Causal language model.
        ids: True prompt tokens.
        mask: Privacy mask.
        observed: Cut activations Eve sees.
        k: Split depth.
        positions: Indices to recover.
        clouds: True-prefix clouds, or ``None``.
        cov: Optional Mahalanobis covariance.
        prior_weight: Log-prior mixture weight.
        score_fn: Optional custom candidate scorer.
        independent_recovery: Use the true prefix at every position.

    Returns:
        SIPIT metric dict.
    """
    extra: dict = {}
    if not independent_recovery:
        extra = {
            "top_k": 80,
            "n_rand": 20,
            "chunk": M.cloud_chunk_size(model),
        }
    return SIP.attack_positions(
        model, ids, mask, observed, k=k, positions=positions,
        cov=cov, prior_weight=prior_weight, clouds=clouds,
        score_fn=score_fn, independent_recovery=independent_recovery,
        **extra,
    )


def _exact_score(
    cloud: torch.Tensor, obs: torch.Tensor, cov: N.GaussianCov,
) -> torch.Tensor:
    """Score candidates by the Gaussian residual likelihood.

    Args:
        cloud: Candidate cut states, ``[B, V', H]``.
        obs: Observed cut state, ``[B, H]``.
        cov: Noise covariance.

    Returns:
        Higher-is-better scores, ``[B, V']``.
    """
    residual = cloud - obs.unsqueeze(1)
    return ED.score_diff_exact(residual, cov, "gaussian")


class _ExactScore:
    """Gaussian exact attacker bound to one covariance."""

    def __init__(self, cov: N.GaussianCov) -> None:
        """Store the covariance for this observation.

        Args:
            cov: Noise covariance on the attack device.
        """
        self.cov = cov

    def __call__(
        self, cloud: torch.Tensor, obs: torch.Tensor,
    ) -> torch.Tensor:
        """Score one candidate cloud.

        Args:
            cloud: Candidate cut states, ``[B, V', H]``.
            obs: Observed cut state, ``[B, H]``.

        Returns:
            Higher-is-better scores, ``[B, V']``.
        """
        return _exact_score(cloud, obs, self.cov)


def _clean_attack_row(clean: dict) -> dict:
    """Build the clean-reference attack row.

    Args:
        clean: Metrics from the noiseless attacker.

    Returns:
        One CSV-ready metrics dict.
    """
    success, total = accuracy_counts(
        clean["token_top1"], clean["n_eval"],
    )
    low, high = wilson_ci(success, total)
    rate = clean["token_top1"]
    return {
        "repeats": 1,
        "repeat_policy": "clean_reference",
        "eve_clean_baseline": rate,
        "eve_vanilla": rate,
        "eve_seq_map": rate,
        "eve_raw_exact": rate,
        "eve_best_attacker": rate,
        "eve_n_eval": clean["n_eval"],
        "wilson_ci_low": low,
        "wilson_ci_high": high,
    }


def _eve_row(
    repeat: int,
    policy: str,
    clean_rate: float,
    vanilla: float,
    sequence: float,
    exact_rate: float,
    n_eval: int,
    ci_low: float,
    ci_high: float,
) -> dict:
    """Build one noisy-attacker metrics row.

    Args:
        repeat: Repeat count.
        policy: Repeat policy name.
        clean_rate: Clean-reference top-1.
        vanilla: Vanilla attacker top-1.
        sequence: Sequence-MAP top-1.
        exact_rate: Exact attacker top-1.
        n_eval: Number of scored tokens.
        ci_low: Wilson interval lower bound.
        ci_high: Wilson interval upper bound.

    Returns:
        One CSV-ready metrics dict.
    """
    return {
        "repeats": int(repeat),
        "repeat_policy": policy,
        "eve_clean_baseline": clean_rate,
        "eve_vanilla": vanilla,
        "eve_seq_map": sequence,
        "eve_raw_exact": exact_rate,
        "eve_best_attacker": max(vanilla, sequence, exact_rate),
        "eve_n_eval": n_eval,
        "wilson_ci_low": ci_low,
        "wilson_ci_high": ci_high,
    }


def _policy_row(
    model: Any,
    ids: torch.Tensor,
    mask: torch.Tensor,
    realizations: list[torch.Tensor],
    cov: N.GaussianCov,
    clouds: dict | None,
    positions: tuple[int, ...],
    device: str,
    dtype: torch.dtype,
    k: int,
    clean: dict,
    repeat: int,
    policy: str,
    independent_recovery: bool,
) -> dict:
    """Score one repeat policy with three attackers.

    Args:
        model: Causal language model.
        ids: True tokens on device.
        mask: Privacy mask on device.
        realizations: Noise realizations of the cut.
        cov: Noise covariance.
        clouds: True-prefix clouds, or ``None``.
        positions: Attack indices.
        device: Torch device.
        dtype: Activation dtype.
        k: Split depth.
        clean: Clean-reference metrics.
        repeat: Repeat count.
        policy: Repeat policy name.
        independent_recovery: Use the true prefix when true.

    Returns:
        One CSV-ready metrics dict.
    """
    observed = observation_for_policy(
        realizations, policy, int(repeat),
    ).to(device, dtype)
    cov_eff = effective_cov_for_policy(cov, policy, int(repeat))
    cov_dev = cov_eff.to(device, dtype)
    shared = (model, ids, mask, observed, k, positions, clouds)
    vanilla = _run_attack(
        *shared, None, 0.0, None, independent_recovery,
    )
    sequence = _run_attack(
        *shared, cov_dev, 1.0, None, independent_recovery,
    )
    exact = _run_attack(
        *shared, None, 0.0, _ExactScore(cov_dev),
        independent_recovery,
    )
    successes, total = accuracy_counts(
        exact["token_top1"], exact["n_eval"],
    )
    ci_low, ci_high = wilson_ci(successes, total)
    return _eve_row(
        int(repeat), policy, clean["token_top1"],
        vanilla["token_top1"], sequence["token_top1"],
        exact["token_top1"], exact["n_eval"], ci_low, ci_high,
    )


def _noisy_attack_rows(
    model: Any,
    ids: torch.Tensor,
    mask: torch.Tensor,
    clipped_a: torch.Tensor,
    cov: N.GaussianCov,
    clouds: dict | None,
    repeats: tuple[int, ...],
    positions: tuple[int, ...],
    seed: int,
    device: str,
    dtype: torch.dtype,
    k: int,
    clean: dict,
    independent_recovery: bool,
) -> list[dict]:
    """Score vanilla, sequence-MAP, and exact attackers.

    Each attacker keeps its own recovered prompt.

    Args:
        model: Causal language model.
        ids: True tokens on device.
        mask: Privacy mask on device.
        clipped_a: Clean cut activations.
        cov: Noise covariance.
        clouds: True-prefix clouds, or ``None``.
        repeats: Repeat counts.
        positions: Attack indices.
        seed: Noise seed.
        device: Torch device.
        dtype: Activation dtype.
        k: Split depth.
        clean: Clean-reference metrics.
        independent_recovery: Use the true prefix when true.

    Returns:
        One row per repeat count and policy.
    """
    k_max = max(max(repeats), 1)
    realizations = materialize_realizations(
        clipped_a, cov, K=k_max, base_seed=seed, dtype=dtype,
    )
    out_rows = []
    for repeat in repeats:
        for policy in REPEAT_POLICIES:
            out_rows.append(_policy_row(
                model, ids, mask, realizations, cov, clouds,
                positions, device, dtype, k, clean, int(repeat),
                policy, independent_recovery,
            ))
    return out_rows


@torch.no_grad()
def attack_metrics(
    model: Any,
    model_id: str,
    k: int,
    cache: dict,
    cov: N.GaussianCov | None,
    repeats: tuple[int, ...],
    n_attack_prompts: int,
    positions: tuple[int, ...],
    seed: int,
    device: str,
    dtype: torch.dtype,
    attack_split: str = "test",
    task: str = "sst2",
    independent_recovery: bool = True,
) -> list[dict]:
    """Score Eve from the true prefix or from each attacker's guesses.

    Args:
        model: Causal language model.
        model_id: Checkpoint id.
        k: Split depth.
        cache: Activation cache for both splits.
        cov: Noise covariance, or ``None`` for the clean reference.
        repeats: Repeat counts.
        n_attack_prompts: How many prompts to attack.
        positions: Token indices to recover.
        seed: Noise seed.
        device: Torch device.
        dtype: Activation dtype.
        attack_split: ``train`` or ``test``.
        task: Benchmark name.
        independent_recovery: Keep the true prefix when true.

    Returns:
        Metric rows for the clean run or each noisy policy.
    """
    attack_cache = cache[attack_split]
    ids = attack_cache["ids"][:n_attack_prompts]
    mask = attack_cache.get(
        "privacy_mask", attack_cache["mask"],
    )[:n_attack_prompts]
    clipped = attack_cache["clipped_a"][:n_attack_prompts]
    clouds = _attack_clouds(
        model, model_id, k, ids, positions, device, task,
        independent_recovery,
    )
    ids_dev = ids.to(device)
    mask_dev = mask.to(device)
    clean = _run_attack(
        model, ids_dev, mask_dev, clipped.to(device, dtype),
        k, positions, clouds, None, 0.0, None, independent_recovery,
    )
    if cov is None:
        return [_clean_attack_row(clean)]
    return _noisy_attack_rows(
        model, ids_dev, mask_dev, clipped, cov, clouds, repeats,
        positions, seed, device, dtype, k, clean, independent_recovery,
    )


def make_skip_row(meta: dict, reason: str) -> dict:
    row = dict(meta)
    row.update({
        "status": "skipped",
        "skip_reason": reason,
        "K": "",
        "repeats": "",
        "repeat_policy": "",
        "kl_no_corr": "",
        "kl_with_corr": "",
        "top1_no_corr": "",
        "top1_with_corr": "",
        "clean_distortion": "",
        "tame_C_sum": "",
        "tame_V_sum": "",
        "margin_cert_noisy": "",
        "ood_maha_mean": "",
        "ood_maha_p95": "",
        "bootstrap_ci_low": "",
        "bootstrap_ci_high": "",
        "eve_clean_baseline": "",
        "eve_vanilla": "",
        "eve_seq_map": "",
        "eve_raw_exact": "",
        "eve_best_attacker": "",
        "eve_n_eval": "",
        "wilson_ci_low": "",
        "wilson_ci_high": "",
    })
    return row


def blank_utility_metrics() -> dict:
    return {
        "kl_no_corr": "",
        "kl_with_corr": "",
        "top1_no_corr": "",
        "top1_with_corr": "",
        "clean_distortion": "",
        "tame_C_sum": "",
        "tame_V_sum": "",
        "margin_cert_noisy": "",
        "ood_maha_mean": "",
        "ood_maha_p95": "",
        "bootstrap_ci_low": "",
        "bootstrap_ci_high": "",
    }


def _parse_bool(text: str) -> bool:
    """Parse a CLI boolean.

    Args:
        text: ``1``/``0``, ``true``/``false``, or ``yes``/``no``.

    Returns:
        Parsed boolean.

    Raises:
        argparse.ArgumentTypeError: If ``text`` is not a boolean.
    """
    lowered = text.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(
        f"expected a boolean, got {text!r}",
    )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--task", default="sst2")
    ap.add_argument("--models", nargs="+", default=["gpt2"])
    ap.add_argument("--ks", nargs="+", type=int, default=[4])
    ap.add_argument("--sfs", nargs="+", type=float, default=[0.5])
    ap.add_argument("--ranks", nargs="+", type=int, default=[16])
    ap.add_argument("--n-attack-prompts", type=int, default=8)
    ap.add_argument("--attack-split", default="test", choices=["train", "test"],
                    help="cache split used for Eve/SIPIT attack metrics")
    ap.add_argument("--positions", nargs="+", type=int, default=None)
    ap.add_argument(
        "--independent_recovery",
        type=_parse_bool,
        default=True,
        help=(
            "1/true scores each position from the true prefix. "
            "0/false feeds each guess into the recovered prompt."
        ),
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--repeats", nargs="+", type=int, default=list(DEFAULT_REPEATS))
    ap.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS),
                    choices=list(DEFAULT_VARIANTS))
    return ap.parse_args()


def _attack_positions(task, cache: dict, split: str, override) -> tuple[int, ...]:
    """Return Eve positions, from the CLI or the task privacy span.

    Args:
        task: Benchmark object.
        cache: Activation cache.
        split: ``train`` or ``test``.
        override: Explicit positions, or ``None``.

    Returns:
        Positions passed to the SIPIT scorer.
    """
    if override:
        return tuple(int(position) for position in override)
    encoded = {
        "input_ids": cache[split]["ids"],
        "privacy_mask": cache[split].get(
            "privacy_mask", cache[split]["mask"],
        ),
    }
    return tuple(task.resolve_privacy_positions(encoded))


def _metric_names(task) -> tuple[str, ...]:
    """Return the utility metric names for a task.

    Args:
        task: Benchmark object.

    Returns:
        Metric ids written into the CSV.
    """
    return tuple(task.utility_metrics)


def _metric_fields(variant: str, scores: dict) -> dict:
    """Flatten utility scores onto one CSV row.

    Args:
        variant: Eval variant name.
        scores: ``metric -> {clean, raw, private}``.

    Returns:
        Column values. Unused channels are left blank.
    """
    fields = {}
    for name, values in scores.items():
        fields[f"{name}_clean"] = values["clean"]
        fields[f"{name}_raw"] = ""
        fields[f"{name}_private"] = ""
        if variant in ("b1_lowrank_struct", "clean_no_noise"):
            fields[f"{name}_raw"] = (
                values["clean"] if variant == "clean_no_noise" else values["raw"]
            )
        if variant in (
            "b1_lowrank_struct_private_suppressor", "clean_no_noise",
        ):
            fields[f"{name}_private"] = (
                values["clean"] if variant == "clean_no_noise"
                else values["private"]
            )
    return fields


def _write_eval_rows(task_name: str, rows: list[dict], run_meta: dict,
                     run_id: str, metric_names: tuple[str, ...]) -> None:
    """Write one eval CSV and JSON per model.

    Args:
        task_name: Benchmark name.
        rows: Eval rows for every model in the run.
        run_meta: Run metadata.
        run_id: Run id used in the JSON filename.
        metric_names: Utility metrics to include as columns.
    """
    models = []
    for row in rows:
        if row["model_id"] not in models:
            models.append(row["model_id"])
    metric_cols = []
    for name in metric_names:
        metric_cols.extend(
            [f"{name}_clean", f"{name}_raw", f"{name}_private"],
        )
    cols = [
        "status", "skip_reason", "model_id", "model_safe", "k",
        "sigma0_frac", "rank", "variant", "K", "repeats", "repeat_policy",
        "seed", "run_id", "git_commit", "A_terms", "attack_split",
        "cov_path", "private_denoiser_path",
        "kl_no_corr", "kl_with_corr", "top1_no_corr", "top1_with_corr",
        "clean_distortion", "tame_C_sum", "tame_V_sum", "margin_cert_noisy",
        "ood_maha_mean", "ood_maha_p95", "bootstrap_ci_low", "bootstrap_ci_high",
        *metric_cols,
        "eve_clean_baseline", "eve_vanilla", "eve_seq_map", "eve_raw_exact",
        "eve_best_attacker", "eve_n_eval", "wilson_ci_low", "wilson_ci_high",
    ]
    for model_id in models:
        model_rows = [row for row in rows if row["model_id"] == model_id]
        out_dir = eval_dir(model_id, task_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "tnsc_eval.csv"
        with open(csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            for row in model_rows:
                writer.writerow({col: row.get(col, "") for col in cols})
        util.write_json(
            out_dir / "tnsc_eval.json",
            {"meta": run_meta, "rows": model_rows},
        )
        util.write_json(
            out_dir / f"{run_id}.json",
            {"meta": run_meta, "rows": model_rows},
        )
        print(f"wrote {csv_path} ({len(model_rows)} rows)", flush=True)


def main():
    args = parse_args()
    # Seed global RNGs + pin deterministic GPU algorithms. Per-run attacker
    # seeds in args.seeds still drive the local generators; this seeds the
    # global state from the primary seed so the whole eval is reproducible.
    seeding.set_seed(args.seeds[0])
    task = get_task(args.task)
    run_meta = util.base_meta(
        Path(__file__).name,
        task=args.task,
        device=args.device,
        models=args.models,
        ks=args.ks,
        sfs=args.sfs,
        ranks=args.ranks,
        n_attack_prompts=args.n_attack_prompts,
        attack_split=args.attack_split,
        positions=args.positions,
        independent_recovery=args.independent_recovery,
        seeds=args.seeds,
        K=args.K,
        repeats=args.repeats,
        variants=args.variants,
        A_terms=list(A_TERMS),
    )
    run_id = run_meta["run_id"]
    rows = []

    for model_id in args.models:
        dtype = M.default_dtype(model_id, args.device)
        model = None
        for k in args.ks:
            cpath = cache_path(model_id, k, task=args.task)
            tw_path = task_weights_path(model_id, k, task=args.task)
            base = {
                "model_id": model_id,
                "model_safe": model_safe(model_id),
                "k": k,
                "run_id": run_id,
                "git_commit": run_meta["git_commit"],
                "A_terms": "+".join(A_TERMS),
                "attack_split": args.attack_split,
            }
            needs_weights = task.kind == "classification"
            if not cpath.exists() or (needs_weights and not tw_path.exists()):
                missing = cpath if not cpath.exists() else tw_path
                reason = f"missing cache/task artifacts: {missing}"
                print(f"SKIP {model_id} k={k}: {reason}", flush=True)
                for sf in args.sfs:
                    for rank in args.ranks:
                        for variant in args.variants:
                            rows.append(make_skip_row({**base, "sigma0_frac": sf, "rank": rank,
                                                       "variant": variant, "seed": ""},
                                                      reason))
                continue
            cache = torch.load(cpath, weights_only=False)
            task_weights = (
                torch.load(tw_path, weights_only=True) if tw_path.exists() else None
            )
            tb_path = task_bias_path(model_id, k, task=args.task)
            task_bias = torch.load(tb_path, weights_only=True) if tb_path.exists() else None
            label_token_ids = None
            meta_path = metadata_path(model_id, k, task=args.task)
            if meta_path.exists() and task.kind == "classification":
                label_token_ids = json.loads(meta_path.read_text()).get("label_token_ids")
            positions = _attack_positions(
                task, cache, args.attack_split, args.positions,
            )
            if model is None:
                model, tokenizer = M.load_model(
                    model_id, dtype=dtype, device=args.device,
                )
            clean_metrics = None
            if task.kind == "generation":
                from src.tasks.generation_eval import _decode, _mean_metrics
                test_split = cache["test"]
                clean_metrics = _mean_metrics(_decode(
                    model, tokenizer, test_split["ids"], test_split["mask"],
                    test_split["clean_a"], k, task.max_new_tokens,
                ), test_split["targets"])
            for sf in args.sfs:
                for rank in args.ranks:
                    for seed in args.seeds:
                        for variant in args.variants:
                            family = VARIANT_TO_COV[variant]
                            meta = {**base, "sigma0_frac": sf, "rank": rank,
                                    "variant": variant, "seed": seed}
                            cov = None
                            cov_file = ""
                            private_state = None
                            private_file = ""
                            row_status = "ok"
                            row_note = ""
                            if family is not None:
                                cov, cov_p = load_cov(model_id, k, sf, rank, family, task=args.task)
                                cov_file = str(cov_p)
                                if cov is None:
                                    rows.append(make_skip_row(meta, f"missing covariance {cov_p}"))
                                    print(f"SKIP {model_id} k={k} sf={sf:g} r={rank} {variant}: missing {cov_p}",
                                          flush=True)
                                    continue
                            private_suffix = None
                            if variant == "b1_lowrank_struct_private_suppressor":
                                private_suffix = "lowrank_struct_private_suppressor"
                            if private_suffix is not None:
                                private_state, private_p = load_private_denoiser_state(
                                    model_id, k, sf, rank, suffix=private_suffix, task=args.task)
                                private_file = str(private_p)
                                if private_state is None:
                                    row_status = "partial"
                                    row_note = f"missing private_denoiser.pt: {private_p}; utility fields empty"
                                    print(f"PARTIAL {model_id} k={k} sf={sf:g} r={rank} {variant}: "
                                          f"{row_note}", flush=True)
                            metric_fields = {}
                            util_metrics = blank_utility_metrics()
                            missing_private = (
                                private_suffix is not None
                                and private_state is None
                            )
                            if not missing_private and task.kind == "generation":
                                test_split = cache["test"]
                                scores = score_generation(
                                    model, tokenizer, test_split["ids"],
                                    test_split["mask"], test_split["clean_a"],
                                    test_split["clipped_a"], test_split["targets"],
                                    k, task.max_new_tokens, cov, private_state,
                                    max(1, args.K), seed, apply_private_denoiser,
                                    clean_metrics=clean_metrics,
                                )
                                metric_fields = _metric_fields(variant, scores)
                            elif not missing_private:
                                scores, util_metrics = utility_metrics(
                                    model, k, cache, task_weights, task_bias,
                                    label_token_ids, cov, private_state,
                                    K=max(1, args.K), seed=seed,
                                    device=args.device, dtype=dtype,
                                )
                                metric_fields = _metric_fields(variant, scores)
                            attack_rows = attack_metrics(
                                model, model_id, k, cache, cov,
                                repeats=tuple(args.repeats),
                                n_attack_prompts=args.n_attack_prompts,
                                positions=positions,
                                seed=seed + 777_000 + 1000 * k,
                                device=args.device, dtype=dtype,
                                attack_split=args.attack_split,
                                task=args.task,
                                independent_recovery=(
                                    args.independent_recovery
                                ),
                            )
                            for arow in attack_rows:
                                row = {
                                    **meta,
                                    "status": row_status,
                                    "skip_reason": row_note,
                                    "K": args.K,
                                    "cov_path": cov_file,
                                    "private_denoiser_path": private_file,
                                    **util_metrics,
                                    **metric_fields,
                                    **arow,
                                }
                                rows.append(row)
                            print(f"{row_status.upper()} {model_id} k={k} sf={sf:g} r={rank} "
                                  f"{variant} seed={seed}",
                                  flush=True)
        if model is not None:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    _write_eval_rows(
        args.task, rows, run_meta, run_id, _metric_names(task),
    )


if __name__ == "__main__":
    main()
