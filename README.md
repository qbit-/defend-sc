# defend-sc — Lowrank-Struct Private Suppressor Frontier

Preliminary experimental code to produce the **privacy / utility
frontier** diagram on Qwen2.5-0.5B / SST-2.

## What the frontier shows

`exact Eve token top-1` (x, teacher-forced attacker recovery under the known noise covariance)
vs `SST-2 accuracy` (y), one point per noise scale, for two channels:
the raw noisy channel and the server-private-suppressed channel. At some regions, the suppressor
recovers task utility while exact Eve stays low. 

## Layout

```
src/                     # transitive import closure (12 modules)
  util.py                #   paths: util.ART -> ./artifacts
  models.py split_model.py sst2_data.py metrics.py
  candidate_sets.py geometry.py sst2_geometry.py
  noise.py               #   GaussianCov: sample / whiten / maha (singular Sigma)
  private_denoise.py     #   PrivateLowrankStructSuppressor (the D_priv being tested)
  attacks/exact_dist.py  #   exact Gaussian log-likelihood scoring (exact Eve)
  attacks/sipit.py       #   per-position candidate-cloud attack (token_top1)
scripts/                 # 7-stage pipeline (run in order; see run_frontier.sh)
run_frontier.sh          # end-to-end driver
```

## Pipeline (each stage writes under `./artifacts/`)

| # | Script | Produces |
|---|--------|----------|
| 1 | `01_collect_calibration.py` | activation cache (train/test clean+clipped), `task_weights` (frozen LM-head rows) |
| 2 | `02_subspace_alignment.py` | task subspace `U_T`, SIPIT-edge subspace `U_S` |
| 3 | `04_build_covariance.py` | base singular `lowrank_struct` covariance `U_s` (rank 8, sf 0.5) |
| 4 | `12b_scale_lowrank_noise.py` | `lowrank_struct` covariances rescaled across the full `sf` menu |
| 5 | `14b_train_private_denoiser.py` | `PrivateLowrankStructSuppressor` per `sf` (fit on train split) |
| 6 | `15_eval_tnsc.py` | `artifacts/setting_g_qwen_sst2_lowrank_private_suppressor/tnsc_eval.csv` |
| 7 | `17_plot_lowrank_private_suppressor.py` | the four plots, incl. the frontier `.png` |

Run it all:

```bash
PY=/path/to/env/python DEVICE=cuda:0 ./run_frontier.sh
```

The frontier lands at
`artifacts/plots/setting_g_qwen_sst2_lowrank_private_suppressor_exact_privacy_utility_frontier.png`.

## Requirements

GPU + PyTorch, `transformers`, `numpy`, `matplotlib`. Validated against the
project `formal-forge` env (PyTorch 2.x, transformers 4.57). Uses SDPA attention
(no flash-attn). Only `gpt2` and `Qwen/Qwen2.5-0.5B` are wired in `src/models.py`.

## Notes on the evaluation (carried over from source)

- Eve is teacher-forced on the true prefix and its candidate set always contains
  the ground-truth token (~101 candidates: top-80 + 20 random + truth), so the
  reported Eve is an oracle-candidate upper bound, not full-vocab reconstruction.
- The suppressor `D_priv` is server-side only; Eve attacks the raw `z = a + eta`.
- Train = SST-2 `train`, test = SST-2 `validation`; the denoiser, noise subspace,
  clip threshold, and task head are all fit on / derived from train (or frozen).
