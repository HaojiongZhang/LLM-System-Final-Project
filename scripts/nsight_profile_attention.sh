#!/usr/bin/env bash
set -euo pipefail

# Generate Nsight Systems and Nsight Compute profiles for the attention matrix.
#
# Usage:
#   bash scripts/nsight_profile_attention.sh
#
# Optional environment overrides:
#   OUT_DIR=results/nsight
#   PROMPT_LENGTHS="128 256"
#   BATCH_SIZE=1
#   NEW_TOKENS=16
#   REPEATS=3
#   MEMORY_REPEATS=1
#   N_VOCAB=512
#   N_EMBD=256
#   N_HEAD=8
#   N_POSITIONS=2048
#   BLOCK_SIZE=16

OUT_DIR="${OUT_DIR:-results/nsight}"
PROMPT_LENGTHS="${PROMPT_LENGTHS:-128 256}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NEW_TOKENS="${NEW_TOKENS:-16}"
REPEATS="${REPEATS:-3}"
MEMORY_REPEATS="${MEMORY_REPEATS:-1}"
N_VOCAB="${N_VOCAB:-512}"
N_EMBD="${N_EMBD:-256}"
N_HEAD="${N_HEAD:-8}"
N_POSITIONS="${N_POSITIONS:-2048}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"

mkdir -p "${OUT_DIR}"

COMMON_ARGS=(
  scripts/benchmark_attention_mode_matrix.py
  --prompt-lengths ${PROMPT_LENGTHS}
  --batch-size "${BATCH_SIZE}"
  --new-tokens "${NEW_TOKENS}"
  --repeats "${REPEATS}"
  --memory-repeats "${MEMORY_REPEATS}"
  --n-vocab "${N_VOCAB}"
  --n-embd "${N_EMBD}"
  --n-head "${N_HEAD}"
  --n-positions "${N_POSITIONS}"
  --block-size "${BLOCK_SIZE}"
  --output-txt "${OUT_DIR}/attention_mode_matrix.txt"
  --output-csv "${OUT_DIR}/attention_mode_matrix.csv"
)

echo "Running Nsight Systems profile..."
nsys profile \
  --force-overwrite true \
  --trace cuda,nvtx,osrt \
  --sample cpu \
  -o "${OUT_DIR}/nsys_attention_matrix" \
  python "${COMMON_ARGS[@]}"

echo "Running Nsight Compute profile..."
ncu \
  --set full \
  --force-overwrite \
  -o "${OUT_DIR}/ncu_attention_matrix" \
  python "${COMMON_ARGS[@]}"

echo "Nsight outputs saved under ${OUT_DIR}"
