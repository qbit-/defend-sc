# Experimental code for securing activation channel against SIPIT attack

Experimental code to produce the **privacy / utility
frontier** diagram on SST-2. The default checkpoint is
Qwen2.5-0.5B. Set `MODEL=Qwen/Qwen3.5-4B` to run the same
pipeline on the Qwen3.5-4B text decoder.

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
scripts/                 # 8-stage pipeline (run in order; see run_frontier.sh)
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
8. `18_measure_inference_slowdown.py`: full-forward vs split, noise, and suppressor time

Run the full metric and plot sequence. Weights use bfloat16 when
CUDA supports it, otherwise float32, and the batch size is 4.
`transformers>=5.5` is required for Qwen3.5.
Change `MODEL` to switch checkpoints. `Qwen/Qwen3.5-4B` loads the
text decoder and drops the unused vision tower.

```bash
export PY=python
export DEVICE=cuda:0
export SEED=0
export MODEL=Qwen/Qwen2.5-0.5B
# export MODEL=Qwen/Qwen3.5-4B
export K=8
export RANK=8
export SFS="0.01 0.02 0.03 0.04 0.05 0.075 0.1 0.125 0.15 0.2 0.25 0.3 0.35 0.4 0.5 0.75 1.0 1.25 1.5 3.0 6.0 10.0"
OUT_TAG="$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]\+/_/g')_sst2_lowrank_private_suppressor"

$PY scripts/01_collect_calibration.py --device "$DEVICE" --models "$MODEL" --ks $K --seed "$SEED"
$PY scripts/02_subspace_alignment.py --device "$DEVICE" --models "$MODEL" --ks $K --seed "$SEED"
$PY scripts/04_build_covariance.py --device "$DEVICE" --models "$MODEL" --ks $K --ranks $RANK --sfs 0.5 --seed "$SEED"
$PY scripts/12b_scale_lowrank_noise.py --models "$MODEL" --ks $K --rank $RANK --families lowrank_struct --base-sf 0.5 --target-sfs $SFS --overwrite --seed "$SEED"
$PY scripts/14b_train_private_denoiser.py --task sst2 --models "$MODEL" --ks $K --ranks $RANK --families lowrank_struct --sfs $SFS --seed "$SEED"
$PY scripts/15_eval_tnsc.py --device "$DEVICE" --task sst2 --models "$MODEL" --ks $K --ranks $RANK --sfs $SFS \
    --variants clean_no_noise b1_lowrank_struct b1_lowrank_struct_private_suppressor \
    --K 2 --repeats 1 --n-attack-prompts 30 --attack-split test \
    --positions 2 5 8 10 12 15 18 20 --seeds "$SEED" \
    --out-tag "$OUT_TAG"
$PY scripts/17_plot_lowrank_private_suppressor.py \
    --csv artifacts/setting_g_${OUT_TAG}/tnsc_eval.csv \
    --prefix setting_g_${OUT_TAG}
$PY scripts/18_measure_inference_slowdown.py --device "$DEVICE" --models "$MODEL" \
    --split-k "$K" --rank "$RANK" --seed "$SEED" \
    --prefix "${OUT_TAG}_inference_slowdown"
```

Or run the wrapper:

```bash
PY=python DEVICE=cuda:0 ./run_frontier.sh
MODEL=Qwen/Qwen3.5-4B PY=python DEVICE=cuda:0 ./run_frontier.sh
```

Metrics land in `artifacts/setting_g_${OUT_TAG}/tnsc_eval.csv`.
The four figures land in `artifacts/plots/`:

- `setting_g_${OUT_TAG}_utility_vs_scale.png`
- `setting_g_${OUT_TAG}_eve_vs_scale.png`
- `setting_g_${OUT_TAG}_exact_privacy_utility_frontier.png`
- `setting_g_${OUT_TAG}_gain_and_clean_distortion.png`
- `${OUT_TAG}_inference_slowdown_bar.png`
- `${OUT_TAG}_inference_slowdown_noise_suppressor_times.png`

Slowdown metrics also land in
`artifacts/inference_slowdown/${OUT_TAG}_inference_slowdown.csv`.
A standalone sweep of the Qwen2.5 sizes plus Qwen3.5-4B is:

```bash
$PY scripts/18_measure_inference_slowdown.py --device "$DEVICE"
```

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