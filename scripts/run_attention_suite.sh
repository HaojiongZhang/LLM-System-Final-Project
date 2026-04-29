#!/usr/bin/env bash

set -u

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python}"
PYTEST_BIN="${PYTEST_BIN:-pytest}"
OUTPUT_FILE="${1:-attention_suite_results.txt}"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$(dirname "$OUTPUT_FILE")"

status=0

run_step() {
  local name="$1"
  shift

  {
    echo
    echo "============================================================"
    echo "$name"
    echo "============================================================"
    echo "Command: $*"
    echo
    "$@"
  } >>"$OUTPUT_FILE" 2>&1

  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    status=1
    {
      echo
      echo "[FAIL] $name (exit code $rc)"
    } >>"$OUTPUT_FILE"
  else
    {
      echo
      echo "[PASS] $name"
    } >>"$OUTPUT_FILE"
  fi
}

{
  echo "Attention Suite Results"
  echo "Generated at: $(date)"
  echo "Repo: $ROOT_DIR"
  echo "Python: $PYTHON_BIN"
  echo "Pytest: $PYTEST_BIN"
} >"$OUTPUT_FILE"

run_step \
  "File Check" \
  bash scripts/check_attention_suite_files.sh

run_step \
  "Syntax Check" \
  "$PYTHON_BIN" -m py_compile \
  scripts/compare_baseline_vs_integrated.py \
  scripts/benchmark_attention_mode_matrix.py \
  scripts/benchmark_fa2_gpu_vs_naive.py \
  tests/test_attention_mode_matrix.py \
  tests/test_flash_attention_func.py \
  tests/test_flash_attention2_backward.py \
  tests/test_flash_attention2_smoke.py \
  tests/test_integration.py

run_step \
  "Attention Tests" \
  "$PYTEST_BIN" -q \
  tests/test_attention_mode_matrix.py \
  tests/test_flash_attention_func.py \
  tests/test_flash_attention2_backward.py \
  tests/test_flash_attention2_smoke.py \
  tests/test_integration.py

run_step \
  "Mode Matrix Benchmark" \
  "$PYTHON_BIN" scripts/benchmark_attention_mode_matrix.py

run_step \
  "Baseline vs Integrated Benchmark" \
  "$PYTHON_BIN" scripts/compare_baseline_vs_integrated.py

run_step \
  "FA2 Benchmark Suite" \
  "$PYTHON_BIN" scripts/benchmark_fa2_gpu_vs_naive.py

{
  echo
  echo "============================================================"
  echo "Final Status"
  echo "============================================================"
  if [[ "$status" -eq 0 ]]; then
    echo "SUCCESS"
  else
    echo "FAILURE"
  fi
} >>"$OUTPUT_FILE"

echo "Wrote results to $OUTPUT_FILE"
exit "$status"
