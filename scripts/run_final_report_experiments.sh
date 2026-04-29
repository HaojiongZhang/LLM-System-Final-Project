#!/usr/bin/env bash
set -euo pipefail

# Main CUDA-machine entry point for final-report evidence.
# This intentionally does not run Nsight; use scripts/nsight_profile_attention.sh
# separately because Nsight runs are slower and produce large files.

OUT_DIR="${OUT_DIR:-results}"
PROMPT_LENGTHS="${PROMPT_LENGTHS:-64 128 256 512}"
N_POSITIONS="${N_POSITIONS:-2048}"
N_VOCAB="${N_VOCAB:-512}"
N_EMBD="${N_EMBD:-256}"
N_HEAD="${N_HEAD:-8}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
mkdir -p "${OUT_DIR}"

# echo "Running correctness tests..."
# pytest -q \
#   tests/test_flash_attention_func.py \
#   tests/test_integration.py \
#   tests/test_scheduler.py \
#   | tee "${OUT_DIR}/final_test_results.txt"

echo "Running attention mode matrix..."
python scripts/benchmark_attention_mode_matrix.py \
  --prompt-lengths ${PROMPT_LENGTHS} \
  --batch-size 1 \
  --new-tokens 16 \
  --repeats 5 \
  --memory-repeats 3 \
  --n-vocab "${N_VOCAB}" \
  --n-embd "${N_EMBD}" \
  --n-head "${N_HEAD}" \
  --n-positions "${N_POSITIONS}" \
  --block-size "${BLOCK_SIZE}" \
  --output-txt "${OUT_DIR}/attention_mode_matrix.txt" \
  --output-csv "${OUT_DIR}/attention_mode_matrix.csv"

echo "Running model size report..."
python scripts/report_model_size.py \
  --n-vocab "${N_VOCAB}" \
  --n-embd "${N_EMBD}" \
  --n-head "${N_HEAD}" \
  --n-positions "${N_POSITIONS}" \
  --seq-len 512 \
  --batch-size 1 \
  --num-blocks 64 \
  --block-size "${BLOCK_SIZE}" \
  --output-json "${OUT_DIR}/model_size.json" \
  | tee "${OUT_DIR}/model_size.txt"

echo "Running dense-vs-flash training loss comparison..."
python scripts/compare_training_loss_flash_vs_dense.py \
  --epochs 3 \
  --samples-per-epoch 1024 \
  --validation-samples 256 \
  --batch-size 16 \
  --model-max-length 40 \
  --n-vocab 4096 \
  --n-embd 128 \
  --n-head 8 \
  --p-dropout 0.0 \
  --output-csv "${OUT_DIR}/training_loss_flash_vs_dense.csv" \
  --output-plot "${OUT_DIR}/training_loss_flash_vs_dense.png" \
  --config-json "${OUT_DIR}/training_loss_flash_vs_dense_config.json"

echo "Plotting final-report figures..."
python scripts/plot_final_report_results.py \
  --attention-csv "${OUT_DIR}/attention_mode_matrix.csv" \
  --training-csv "${OUT_DIR}/training_loss_flash_vs_dense.csv" \
  --output-dir "${OUT_DIR}/plots"

echo "Writing markdown tables..."
python scripts/make_report_tables.py \
  --attention-csv "${OUT_DIR}/attention_mode_matrix.csv" \
  --model-size-json "${OUT_DIR}/model_size.json" \
  --output-md "${OUT_DIR}/final_report_tables.md"

echo "Final-report experiment outputs saved under ${OUT_DIR}"
