# Experimental code for securing activation channel against SIPIT attack

Experimental code to produce the **privacy / utility
frontier** diagram on Qwen2.5-0.5B on the SST-2 benchmark\.

## What the frontier shows

`exact Eve token top-1` (x, teacher-forced attacker recovery under the known noise covariance)
vs `SST-2 accuracy` (y), one point per noise scale, for two channels:
the raw noisy channel and the server-private-suppressed channel. At some regions, the suppressor can recover some task utility while exact Eve stays low. 

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

Find a detailed overview of the code and the method in [Overview](./docs/OVERVIEW.md)

## Pipeline (each stage writes under `./artifacts/`)

1. `01_collect_calibration.py`: activation cache (train/test clean+clipped), `task_weights` (frozen LM-head rows) 
2. `02_subspace_alignment.py`: task subspace `U_T`, SIPIT-edge subspace `U_S`
3. `04_build_covariance.py`: base singular `lowrank_struct` covariance `U_s`
4. `12b_scale_lowrank_noise.py`: `lowrank_struct` covariances rescaled across noise scales
5. `14b_train_private_denoiser.py`: `PrivateLowrankStructSuppressor` per noise scale (fit on train split)
6. `15_eval_tnsc.py`: evaluate exact Eve and SST-2 accuracy across noise scales, with and without the suppressor; write `tnsc_eval.csv`
7. `17_plot_lowrank_private_suppressor.py`: the four plots, incl. the frontier `.png

Run it all:

```bash
PY=/path/to/env/python DEVICE=cuda:0 ./run_frontier.sh
```

The figures land at
`artifacts/plots/`.

## Exporting Markdown documents to PDF

On a SageMaker Ubuntu notebook, install XeLaTeX and fonts
(Pandoc is typically already present):

```bash
sudo apt-get update
sudo apt-get install -y \
  texlive-xetex \
  texlive-fonts-recommended \
  texlive-lang-cyrillic \
  lmodern \
  fonts-dejavu
```

Run Pandoc:

```bash
pandoc docs/OVERVIEW_full_rus.md \
  -o OVERVIEW_full_rus.pdf \
  --pdf-engine=xelatex \
  -V lang=ru \   # For Russian texts, drop this for English
  -V mainfont="DejaVu Serif" \
  -V sansfont="DejaVu Sans" \
  -V monofont="DejaVu Sans Mono" \
  -V geometry:margin=1in \
  --resource-path=.:docs
```

`--resource-path` lets relative plot links resolve.

## Notes on the evaluation

- Eve is teacher-forced on the true prefix and its candidate set always contains the ground-truth token (~101 candidates: top-80 + 20 random + truth), so the reported Eve success rate is an oracle-candidate upper bound, not full-vocab reconstruction.
- Train = SST-2 `train`, test = SST-2 `validation`; the denoiser, noise subspace, clip threshold, and task head are all fit on / derived from train (or frozen).
- SIPIT paper's gradient-based SIPIT method have not been evaluated