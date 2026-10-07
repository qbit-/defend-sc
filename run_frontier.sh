#!/usr/bin/env bash
# Reproduce the one-shot privacy/utility frontier for the lowrank_struct
# server-private suppressor (split k=8, rank 8).
#
#   MODEL=Qwen/Qwen2.5-0.5B ./run_frontier.sh
#   MODEL=Qwen/Qwen3.5-4B TASK=word_sorting ./run_frontier.sh
#
# Requires a GPU and the project env (PyTorch, transformers>=5.5).
# Uses .venv/bin/python when PY is unset and that interpreter exists.
set -euo pipefail

cd "$(dirname "$0")"
if [[ -z "${PY:-}" && -x .venv/bin/python ]]; then
    PY=".venv/bin/python"
else
    PY="${PY:-python}"
fi
DEVICE="${DEVICE:-cuda:0}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
TASK="${TASK:-sst2}"
K="${K:-8}"
RANK="${RANK:-8}"
SEED="${SEED:-0}"
# Eve prefix: 1 = true tokens, 0 = attacker's own guesses.
INDEPENDENT_RECOVERY="${INDEPENDENT_RECOVERY:-1}"
SFS="0.01 0.02 0.03 0.04 0.05 0.075 0.1 0.125 0.15 0.2 0.25 0.3 0.35 0.4 0.5 0.75 1.0 1.25 1.5 3.0 6.0 10.0"
SAFE_MODEL="${MODEL//\//_}"
PREFIX="lowrank_private_suppressor"
# Privacy offsets for Eve. An empty OFFSET_PATTERN makes POSITIONS
# absolute token indices on every task. A pattern applies those
# offsets after the last match of that text. SST-2's default grid
# is filled in only when neither variable is set.
if [[ -z "${POSITIONS+x}" && -z "${OFFSET_PATTERN+x}" && "$TASK" == "sst2" ]]; then
    POSITIONS="2 5 8 10 12 15 18 20"
fi
POSITIONS="${POSITIONS:-}"
OFFSET_PATTERN="${OFFSET_PATTERN:-}"
POSITION_ARGS=()
if [[ -n "$POSITIONS" ]]; then
    # Word-splitting is intentional: each offset is one argument.
    # shellcheck disable=SC2086
    POSITION_ARGS+=(--positions $POSITIONS)
fi
if [[ -n "$OFFSET_PATTERN" ]]; then
    POSITION_ARGS+=(--offset_pattern "$OFFSET_PATTERN")
fi

# Single reproducibility knob. SEED is passed to every stage, where
# src/seeding.set_seed() seeds the global Python/NumPy/Torch/CUDA RNGs and pins
# deterministic GPU algorithms (cuDNN deterministic, TF32 off). Export
# PYTHONHASHSEED before the interpreter starts so hash randomization is pinned
# for all stages. Set DEFEND_SC_STRICT_DETERMINISM=1 to additionally request
# deterministic CUDA kernels (slower; warns rather than fails on unsupported ops).
export PYTHONHASHSEED="$SEED"

# Keep Hugging Face downloads in a writable project-local cache unless the user
# has already configured a writable cache location.
cache_writable() {
    local path="$1"
    local parent="$path"
    if [[ -e "$path" ]]; then
        [[ -w "$path" ]]
        return
    fi
    while [[ ! -e "$parent" && "$parent" != "/" ]]; do
        parent="$(dirname "$parent")"
    done
    [[ -w "$parent" ]]
}

DEFAULT_HF_HOME="$PWD/.hf_cache"
if [[ -z "${HF_HOME:-}" ]] || ! cache_writable "$HF_HOME"; then
    export HF_HOME="$DEFAULT_HF_HOME"
fi
if [[ -z "${HF_HUB_CACHE:-}" ]] || ! cache_writable "$HF_HUB_CACHE"; then
    export HF_HUB_CACHE="$HF_HOME/hub"
fi
if [[ -z "${TRANSFORMERS_CACHE:-}" ]] || ! cache_writable "$TRANSFORMERS_CACHE"; then
    export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
fi

# Scaffolding dirs the source scripts assume exist (committed in the original repo).
mkdir -p artifacts/reports artifacts/plots "$HF_HOME" "$HF_HUB_CACHE"

echo "Model: $MODEL"
echo "Task: $TASK"
echo "Attack offsets: ${POSITIONS:-task default}"
echo "Offset pattern: ${OFFSET_PATTERN:-<absolute or task span>}"
echo "Independent recovery: $INDEPENDENT_RECOVERY"
echo "Outputs: artifacts/evals/${SAFE_MODEL}/${TASK}/ and artifacts/plots/${SAFE_MODEL}/${TASK}/"

# 1. Cache clean + clipped cut activations, answer-position features, task head (LM-head rows).
# $PY scripts/01_collect_calibration.py --device "$DEVICE" --task "$TASK" --models "$MODEL" --ks $K --seed "$SEED"

# 2. Phase-2 geometry: task subspace U_T + SIPIT-edge subspace U_S (writes geometry/.../U_T.pt).
# $PY scripts/02_subspace_alignment.py --device "$DEVICE" --task "$TASK" --models "$MODEL" --ks $K --seed "$SEED"

# 3. Build the rank-8 base covariance incl. the singular lowrank_struct subspace U_s.
#    Only the base sf=0.5 is needed here; step 4 rescales it to the full sf menu.
# $PY scripts/04_build_covariance.py --device "$DEVICE" --task "$TASK" --models "$MODEL" --ks $K \
#     --ranks $RANK --sfs 0.5 --seed "$SEED"

# 4. Re-scale the lowrank_struct covariance across the full sf menu (reuses U_s
#    orientation). Note 12b's flags: --rank (singular) and --target-sfs.
# $PY scripts/12b_scale_lowrank_noise.py --task "$TASK" --models "$MODEL" --ks $K --rank $RANK \
#     --families lowrank_struct --base-sf 0.5 --target-sfs $SFS --overwrite --seed "$SEED"

# 5. Fit the server-private Wiener suppressor D_priv per sf (train split only).
# $PY scripts/14b_train_private_denoiser.py --task "$TASK" --models "$MODEL" --ks $K --ranks $RANK \
#     --families lowrank_struct --sfs $SFS --seed "$SEED"

# 6. Evaluate utility (raw vs private) and Eve (vanilla / exact / seq-MAP) -> tnsc_eval.csv.
$PY scripts/15_eval_tnsc.py --device "$DEVICE" --task "$TASK" --models "$MODEL" --ks $K --ranks $RANK \
    --sfs $SFS \
    --variants clean_no_noise b1_lowrank_struct b1_lowrank_struct_private_suppressor \
    --K 2 --repeats 1 --n-attack-prompts 30 --attack-split test \
    ${POSITION_ARGS+"${POSITION_ARGS[@]}"} \
    --independent_recovery "$INDEPENDENT_RECOVERY" \
    --seeds "$SEED"

# 7. Render Eve plus one utility/frontier/gain figure per metric.
$PY scripts/17_plot_lowrank_private_suppressor.py \
    --csv "artifacts/evals/${SAFE_MODEL}/${TASK}/tnsc_eval.csv" \
    --prefix "$PREFIX"

# 8. Time full inference against the split, noise, and suppressor.
$PY scripts/18_measure_inference_slowdown.py \
    --device "$DEVICE" --task "$TASK" --models "$MODEL" --split-k "$K" --rank "$RANK" \
    --seed "$SEED" --prefix "inference_slowdown"

echo
echo "Metrics: artifacts/evals/${SAFE_MODEL}/${TASK}/tnsc_eval.csv"
echo "Plots: artifacts/plots/${SAFE_MODEL}/${TASK}/"
