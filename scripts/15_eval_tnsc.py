"""Setting G TNSC evaluation scaffold for SST-2/logit pilot artifacts."""
from __future__ import annotations
import os, sys, argparse, json, csv
from pathlib import Path
import math
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/claude-hf-cache")

from src import candidate_sets as CS
from src import metrics as MET
from src import models as M
from src import noise as N
from src import split_model as SM
from src import sst2_data as D
from src import util
from src import seeding
from src.attacks import exact_dist as ED
from src.attacks import sipit as SIP


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
    return model_id.replace("/", "_")


def load_task_module(task: str):
    if task == "sst2":
        from src import sst2_data as task_data
    else:
        raise ValueError(f"unknown task {task!r}")
    return task_data


def cache_path(model_id: str, k: int, task: str = "sst2") -> Path:
    return util.ART / "activations" / model_safe(model_id) / task / f"split_{k}" / "cache.pt"


def task_weights_path(model_id: str, k: int, task: str = "sst2") -> Path:
    return util.ART / "activations" / model_safe(model_id) / task / f"split_{k}" / "task_weights.pt"


def task_bias_path(model_id: str, k: int, task: str = "sst2") -> Path:
    return util.ART / "activations" / model_safe(model_id) / task / f"split_{k}" / "task_bias.pt"


def metadata_path(model_id: str, k: int, task: str = "sst2") -> Path:
    return util.ART / "activations" / model_safe(model_id) / task / f"split_{k}" / "metadata.json"


def cov_path(model_id: str, k: int, sigma0_frac: float, rank: int, family: str,
             task: str = "sst2") -> Path:
    return (util.ART / "covariances" / model_safe(model_id) / task / f"split_{k}"
            / f"sigma0_{sigma0_frac:g}_r{rank}__{family}.pt")


def private_denoiser_path(model_id: str, k: int, sigma0_frac: float, rank: int,
                          suffix: str = "tnsc_gaussian", task: str = "sst2") -> Path:
    return (util.ART / "private_denoisers" / model_safe(model_id) / task / f"split_{k}"
            / f"sigma0_{sigma0_frac:g}_r{rank}__{suffix}" / "private_denoiser.pt")


def cloud_cache_dir(model_id: str, k: int) -> Path:
    return util.ART / "sipit_clouds" / model_safe(model_id) / f"split_{k}"


def setting_g_dir() -> Path:
    return util.ART / "setting_g"


def setting_g_task_dir(task: str, out_tag: str = "") -> Path:
    if out_tag:
        return util.ART / f"setting_g_{out_tag}"
    return util.ART / ("setting_g" if task == "sst2" else f"setting_g_{task}")


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
                         chunk: int, device) -> dict:
    out_dir = cloud_cache_dir(model_id, k)
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
    vocab = model.config.vocab_size
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
def server_logits_for(model, k: int, ids, mask, ans_pos, hidden, task_weights, device, dtype,
                      task_bias=None, label_token_ids=None):
    out = SM.split_run(model, ids.to(device), k=k, hidden_override=hidden.to(device, dtype),
                       attention_mask=mask.to(device))
    if label_token_ids is not None:
        pos = ans_pos.to(device).view(-1, 1, 1).expand(-1, 1, out["logits"].shape[-1])
        last = out["logits"].gather(1, pos).squeeze(1)
        label_ids = torch.tensor(label_token_ids, device=device)
        return last.index_select(-1, label_ids).cpu().float()
    feat = D.gather_answer_position(out["server_out"], ans_pos.to(device)).cpu().float()
    logits = feat @ task_weights.float().T
    if task_bias is not None:
        logits = logits + task_bias.float()
    return logits


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
        return {
            "acc_clean_test": acc_clean,
            "acc_no_corr": acc_clean,
            "acc_with_corr": acc_clean,
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
    return {
        "acc_clean_test": acc_clean,
        "acc_no_corr": MET.accuracy(mean_no, labels),
        "acc_with_corr": MET.accuracy(mean_corr, labels),
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


@torch.no_grad()
def attack_metrics(model, model_id: str, k: int, cache: dict, cov: N.GaussianCov | None,
                   repeats: tuple[int, ...],
                   n_attack_prompts: int, positions: tuple[int, ...],
                   seed: int, device, dtype, attack_split: str = "test") -> list[dict]:
    attack_cache = cache[attack_split]
    ids = attack_cache["ids"][:n_attack_prompts]
    mask = attack_cache["mask"][:n_attack_prompts]
    clipped_a = attack_cache["clipped_a"][:n_attack_prompts]
    clouds = build_or_load_clouds(model, model_id, k, ids, positions, top_k=80,
                                  n_rand=20, chunk=256, device=device)

    ids_dev = ids.to(device)
    mask_dev = mask.to(device)
    clean = SIP.attack_positions(model, ids_dev, mask_dev, clipped_a.to(device, dtype),
                                 k=k, positions=positions, cov=None, prior_weight=0.0,
                                 clouds=clouds)
    clean_success, clean_total = accuracy_counts(clean["token_top1"], clean["n_eval"])
    clean_ci = wilson_ci(clean_success, clean_total)

    if cov is None:
        return [{
            "repeats": 1,
            "repeat_policy": "clean_reference",
            "eve_clean_baseline": clean["token_top1"],
            "eve_vanilla": clean["token_top1"],
            "eve_seq_map": clean["token_top1"],
            "eve_raw_exact": clean["token_top1"],
            "eve_best_attacker": clean["token_top1"],
            "eve_n_eval": clean["n_eval"],
            "wilson_ci_low": clean_ci[0],
            "wilson_ci_high": clean_ci[1],
        }]

    K_max = max(max(repeats), 1)
    realizations = materialize_realizations(clipped_a, cov, K=K_max, base_seed=seed, dtype=dtype)
    out_rows = []
    for m in repeats:
        for policy in REPEAT_POLICIES:
            a_bar = observation_for_policy(realizations, policy, int(m)).to(device, dtype)
            cov_eff = effective_cov_for_policy(cov, policy, int(m))
            cov_eff_dev = cov_eff.to(device, dtype)
            van = SIP.attack_positions(model, ids_dev, mask_dev, a_bar, k=k,
                                       positions=positions, cov=None, prior_weight=0.0,
                                       clouds=clouds)
            seq = SIP.attack_positions(model, ids_dev, mask_dev, a_bar, k=k,
                                       positions=positions, cov=cov_eff, prior_weight=1.0,
                                       clouds=clouds)

            def exact_score(cloud, obs):
                return ED.score_diff_exact(cloud - obs.unsqueeze(1), cov_eff_dev, "gaussian")

            exact = SIP.attack_positions(model, ids_dev, mask_dev, a_bar, k=k,
                                         positions=positions, cov=None, prior_weight=0.0,
                                         clouds=clouds, score_fn=exact_score)

            successes, total = accuracy_counts(exact["token_top1"], exact["n_eval"])
            ci_low, ci_high = wilson_ci(successes, total)
            vals = [van["token_top1"], seq["token_top1"], exact["token_top1"]]
            out_rows.append({
                "repeats": int(m),
                "repeat_policy": policy,
                "eve_clean_baseline": clean["token_top1"],
                "eve_vanilla": van["token_top1"],
                "eve_seq_map": seq["token_top1"],
                "eve_raw_exact": exact["token_top1"],
                "eve_best_attacker": max(vals),
                "eve_n_eval": exact["n_eval"],
                "wilson_ci_low": ci_low,
                "wilson_ci_high": ci_high,
            })
    return out_rows


def make_skip_row(meta: dict, reason: str) -> dict:
    row = dict(meta)
    row.update({
        "status": "skipped",
        "skip_reason": reason,
        "K": "",
        "repeats": "",
        "repeat_policy": "",
        "acc_clean_test": "",
        "acc_no_corr": "",
        "acc_with_corr": "",
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
        "acc_clean_test": "",
        "acc_no_corr": "",
        "acc_with_corr": "",
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--task", default="sst2", choices=["sst2"])
    ap.add_argument("--models", nargs="+", default=["gpt2"])
    ap.add_argument("--ks", nargs="+", type=int, default=[4])
    ap.add_argument("--sfs", nargs="+", type=float, default=[0.5])
    ap.add_argument("--ranks", nargs="+", type=int, default=[16])
    ap.add_argument("--n-attack-prompts", type=int, default=8)
    ap.add_argument("--attack-split", default="test", choices=["train", "test"],
                    help="cache split used for Eve/SIPIT attack metrics")
    ap.add_argument("--positions", nargs="+", type=int, default=[2, 5, 10, 20])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--repeats", nargs="+", type=int, default=list(DEFAULT_REPEATS))
    ap.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS),
                    choices=list(DEFAULT_VARIANTS))
    ap.add_argument("--out-tag", default="",
                    help="optional output directory suffix: artifacts/setting_g_{out_tag}")
    return ap.parse_args()


def main():
    args = parse_args()
    # Seed global RNGs + pin deterministic GPU algorithms. Per-run attacker
    # seeds in args.seeds still drive the local generators; this seeds the
    # global state from the primary seed so the whole eval is reproducible.
    seeding.set_seed(args.seeds[0])
    global D
    D = load_task_module(args.task)
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
        seeds=args.seeds,
        K=args.K,
        repeats=args.repeats,
        variants=args.variants,
        out_tag=args.out_tag,
        A_terms=list(A_TERMS),
    )
    run_id = run_meta["run_id"]
    rows = []
    dtype = torch.float32

    for model_id in args.models:
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
            if not cpath.exists() or not tw_path.exists():
                reason = f"missing cache/task artifacts: {cpath if not cpath.exists() else tw_path}"
                print(f"SKIP {model_id} k={k}: {reason}", flush=True)
                for sf in args.sfs:
                    for rank in args.ranks:
                        for variant in args.variants:
                            rows.append(make_skip_row({**base, "sigma0_frac": sf, "rank": rank,
                                                       "variant": variant, "seed": ""},
                                                      reason))
                continue
            cache = torch.load(cpath, weights_only=False)
            task_weights = torch.load(tw_path, weights_only=True)
            tb_path = task_bias_path(model_id, k, task=args.task)
            task_bias = torch.load(tb_path, weights_only=True) if tb_path.exists() else None
            label_token_ids = None
            meta_path = metadata_path(model_id, k, task=args.task)
            if args.task == "sst2" and meta_path.exists():
                label_token_ids = json.loads(meta_path.read_text()).get("label_token_ids")
            if model is None:
                model, _ = M.load_model(model_id, dtype=dtype, device=args.device)
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
                            if private_suffix is not None and private_state is None:
                                util_metrics = blank_utility_metrics()
                            else:
                                util_metrics = utility_metrics(
                                    model, k, cache, task_weights, task_bias, label_token_ids, cov,
                                    private_state, K=max(1, args.K), seed=seed,
                                    device=args.device, dtype=dtype)
                            attack_rows = attack_metrics(
                                model, model_id, k, cache, cov,
                                repeats=tuple(args.repeats),
                                n_attack_prompts=args.n_attack_prompts,
                                positions=tuple(args.positions),
                                seed=seed + 777_000 + 1000 * k,
                                device=args.device, dtype=dtype,
                                attack_split=args.attack_split)
                            for arow in attack_rows:
                                row = {
                                    **meta,
                                    "status": row_status,
                                    "skip_reason": row_note,
                                    "K": args.K,
                                    "cov_path": cov_file,
                                    "private_denoiser_path": private_file,
                                    **util_metrics,
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

    out_dir = setting_g_task_dir(args.task, args.out_tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [
        "status", "skip_reason", "model_id", "model_safe", "k", "sigma0_frac", "rank",
        "variant", "K", "repeats", "repeat_policy", "seed", "run_id", "git_commit",
        "A_terms", "attack_split", "cov_path", "private_denoiser_path",
        "acc_clean_test", "acc_no_corr",
        "acc_with_corr", "kl_no_corr", "kl_with_corr", "top1_no_corr", "top1_with_corr",
        "clean_distortion", "tame_C_sum", "tame_V_sum", "margin_cert_noisy",
        "ood_maha_mean", "ood_maha_p95", "bootstrap_ci_low", "bootstrap_ci_high",
        "eve_clean_baseline", "eve_vanilla",
        "eve_seq_map", "eve_raw_exact",
        "eve_best_attacker", "eve_n_eval", "wilson_ci_low",
        "wilson_ci_high",
    ]
    csv_path = out_dir / "tnsc_eval.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in cols})
    util.write_json(out_dir / "tnsc_eval.json", {"meta": run_meta, "rows": rows})
    util.write_json(out_dir / f"{run_id}.json", {"meta": run_meta, "rows": rows})
    print(f"wrote {csv_path} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
