#!/usr/bin/env bash
# Reproduce the one-shot privacy/utility frontier diagram for the lowrank_struct
# server-private suppressor (Qwen2.5-0.5B, SST-2, split k=8, rank 8).
#
# Output plot: artifacts/plots/setting_g_qwen_sst2_lowrank_private_suppressor_exact_privacy_utility_frontier.png
#
# Requires a GPU + the project env (PyTorch, transformers). Set PY to its python.
set -euo pipefail

PY="${PY:-/Users/qinghuazhou/mambaforge/envs/formal-forge/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
MODEL="Qwen/Qwen2.5-0.5B"
K=8
RANK=8
SEED="${SEED:-0}"
SFS="0.01 0.02 0.03 0.04 0.05 0.075 0.1 0.125 0.15 0.2 0.25 0.3 0.35 0.4 0.5 0.75 1.0 1.25 1.5 3.0 6.0 10.0"

# Single reproducibility knob. SEED is passed to every stage, where
# src/seeding.set_seed() seeds the global Python/NumPy/Torch/CUDA RNGs and pins
# deterministic GPU algorithms (cuDNN deterministic, TF32 off). Export
# PYTHONHASHSEED before the interpreter starts so hash randomization is pinned
# for all stages. Set DEFEND_SC_STRICT_DETERMINISM=1 to additionally request
# deterministic CUDA kernels (slower; warns rather than fails on unsupported ops).
export PYTHONHASHSEED="$SEED"

cd "$(dirname "$0")"

# Scaffolding dirs the source scripts assume exist (committed in the original repo).
mkdir -p artifacts/reports artifacts/plots

# 1. Cache clean + clipped cut activations, answer-position features, task head (LM-head rows).
$PY scripts/01_collect_calibration.py --device "$DEVICE" --models "$MODEL" --ks $K --seed "$SEED"

# 2. Phase-2 geometry: task subspace U_T + SIPIT-edge subspace U_S (writes geometry/.../U_T.pt).
$PY scripts/02_subspace_alignment.py --device "$DEVICE" --models "$MODEL" --ks $K --seed "$SEED"

# 3. Build the rank-8 base covariance incl. the singular lowrank_struct subspace U_s.
#    Only the base sf=0.5 is needed here; step 4 rescales it to the full sf menu.
$PY scripts/04_build_covariance.py --device "$DEVICE" --models "$MODEL" --ks $K \
    --ranks $RANK --sfs 0.5 --seed "$SEED"

# 4. Re-scale the lowrank_struct covariance across the full sf menu (reuses U_s
#    orientation). Note 12b's flags: --rank (singular) and --target-sfs.
$PY scripts/12b_scale_lowrank_noise.py --models "$MODEL" --ks $K --rank $RANK \
    --families lowrank_struct --base-sf 0.5 --target-sfs $SFS --overwrite --seed "$SEED"

# 5. Fit the server-private Wiener suppressor D_priv per sf (train split only).
$PY scripts/14b_train_private_denoiser.py --task sst2 --models "$MODEL" --ks $K --ranks $RANK \
    --families lowrank_struct --sfs $SFS --seed "$SEED"

# 6. Evaluate utility (raw vs private) and Eve (vanilla / exact / seq-MAP) -> tnsc_eval.csv.
$PY scripts/15_eval_tnsc.py --device "$DEVICE" --task sst2 --models "$MODEL" --ks $K --ranks $RANK \
    --sfs $SFS \
    --variants clean_no_noise b1_lowrank_struct b1_lowrank_struct_private_suppressor \
    --K 2 --repeats 1 --n-attack-prompts 30 --attack-split test \
    --positions 2 5 8 10 12 15 18 20 --seeds "$SEED" \
    --out-tag qwen_sst2_lowrank_private_suppressor

# 7. Render the four plots, including the privacy/utility frontier.
$PY scripts/17_plot_lowrank_private_suppressor.py \
    --csv artifacts/setting_g_qwen_sst2_lowrank_private_suppressor/tnsc_eval.csv \
    --prefix setting_g_qwen_sst2_lowrank_private_suppressor

echo
echo "Frontier diagram:"
echo "  artifacts/plots/setting_g_qwen_sst2_lowrank_private_suppressor_exact_privacy_utility_frontier.png"
