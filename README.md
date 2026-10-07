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
src/                     # shared model, attack, and benchmark code
  util.py                #   paths: util.ART -> ./artifacts
  models.py split_model.py generate.py sst2_data.py metrics.py
  candidate_sets.py geometry.py sst2_geometry.py
  tasks/                 #   sst2 and word_sorting adapters
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
6. `15_eval_tnsc.py`: evaluate exact Eve and task utility across noise scales, with and without the suppressor; write `tnsc_eval.csv`
7. `17_plot_lowrank_private_suppressor.py`: Eve, plus one utility, frontier, and gain plot per metric
8. `18_measure_inference_slowdown.py`: full-forward vs split, noise, and suppressor time

Benchmarks are selected with `TASK`. `sst2` reports accuracy. `word_sorting` is BBH word sorting (fit on the larger BIG-bench set, scored on all 250 BBH examples) and reports exact match and character edit similarity on separate plots. Artifacts are grouped as `artifacts/<kind>/<model>/<task>/`.

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
export TASK=sst2
# export TASK=word_sorting
# export MODEL=Qwen/Qwen3.5-4B
export K=8
export RANK=8
export SFS="0.01 0.02 0.03 0.04 0.05 0.075 0.1 0.125 0.15 0.2 0.25 0.3 0.35 0.4 0.5 0.75 1.0 1.25 1.5 3.0 6.0 10.0"
SAFE_MODEL="${MODEL//\//_}"
# Empty OFFSET_PATTERN: POSITIONS are absolute token indices.
# A pattern counts those offsets after its last match.
if [[ -z "${POSITIONS+x}" && -z "${OFFSET_PATTERN+x}" && "$TASK" == "sst2" ]]; then
  POSITIONS="2 5 8 10 12 15 18 20"
fi
POSITIONS="${POSITIONS:-}"
OFFSET_PATTERN="${OFFSET_PATTERN:-}"
POSITION_ARGS=()
if [[ -n "$POSITIONS" ]]; then
  # shellcheck disable=SC2086
  POSITION_ARGS+=(--positions $POSITIONS)
fi
if [[ -n "$OFFSET_PATTERN" ]]; then
  POSITION_ARGS+=(--offset_pattern "$OFFSET_PATTERN")
fi

$PY scripts/01_collect_calibration.py --device "$DEVICE" --task "$TASK" --models "$MODEL" --ks $K --seed "$SEED"
$PY scripts/02_subspace_alignment.py --device "$DEVICE" --task "$TASK" --models "$MODEL" --ks $K --seed "$SEED"
$PY scripts/04_build_covariance.py --device "$DEVICE" --task "$TASK" --models "$MODEL" --ks $K --ranks $RANK --sfs 0.5 --seed "$SEED"
$PY scripts/12b_scale_lowrank_noise.py --task "$TASK" --models "$MODEL" --ks $K --rank $RANK --families lowrank_struct --base-sf 0.5 --target-sfs $SFS --overwrite --seed "$SEED"
$PY scripts/14b_train_private_denoiser.py --task "$TASK" --models "$MODEL" --ks $K --ranks $RANK --families lowrank_struct --sfs $SFS --seed "$SEED"
$PY scripts/15_eval_tnsc.py --device "$DEVICE" --task "$TASK" --models "$MODEL" --ks $K --ranks $RANK --sfs $SFS \
    --variants clean_no_noise b1_lowrank_struct b1_lowrank_struct_private_suppressor \
    --K 2 --repeats 1 --n-attack-prompts 30 --attack-split test \
    ${POSITION_ARGS+"${POSITION_ARGS[@]}"} --seeds "$SEED"
$PY scripts/17_plot_lowrank_private_suppressor.py \
    --csv artifacts/evals/${SAFE_MODEL}/${TASK}/tnsc_eval.csv \
    --prefix lowrank_private_suppressor
$PY scripts/18_measure_inference_slowdown.py --device "$DEVICE" --task "$TASK" --models "$MODEL" \
    --split-k "$K" --rank "$RANK" --seed "$SEED" \
    --prefix inference_slowdown
```

Or run the wrapper:

```bash
PY=python DEVICE=cuda:0 ./run_frontier.sh
TASK=word_sorting MODEL=Qwen/Qwen3.5-4B PY=python DEVICE=cuda:0 ./run_frontier.sh
```

Metrics land in `artifacts/evals/<model>/<task>/tnsc_eval.csv`.
Figures land in `artifacts/plots/<model>/<task>/`:

- `lowrank_private_suppressor_eve_vs_scale.png`
- `lowrank_private_suppressor_utility_vs_scale_<metric>.png`
- `lowrank_private_suppressor_frontier_<metric>.png`
- `lowrank_private_suppressor_gain_<metric>.png`
- `inference_slowdown_bar.png`
- `inference_slowdown_noise_suppressor_times.png`

`<metric>` is `accuracy` for SST-2, and `exact_match` plus
`char_edit_similarity` for word sorting. Slowdown metrics also land in
`artifacts/inference_slowdown/<model>/<task>/inference_slowdown.csv`.
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
- For SST-2, train is GLUE `train` and test is GLUE `validation`. For word sorting, train is BIG-bench with the 250 BBH targets removed, and test is those 250 BBH examples. The denoiser, noise subspace, and clip threshold are fit on train.
- SIPIT paper's gradient-based SIPIT method have not been evaluated