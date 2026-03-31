#!/bin/bash
#SBATCH --job-name=fa2-gpu-bench
#SBATCH -p GPU-shared
#SBATCH --gpus=v100:1
#SBATCH -t 01:15:00
#SBATCH --mem=16G
#SBATCH -o /jet/home/kchien/LLM-System-Final-Project-/fa2_gpu_bench_%j.log

set -euo pipefail

cd /jet/home/kchien/LLM-System-Final-Project-

echo "[INFO] Job ID: ${SLURM_JOB_ID}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Start: $(date)"

module load cuda
source /jet/home/kchien/llmsys_hw3/.venv/bin/activate

echo "[INFO] Python: $(which python)"
echo "[INFO] CUDA:"
nvcc --version | head -n 4 || true
echo "[INFO] nvidia-smi:"
nvidia-smi || true

echo "[INFO] Building CUDA kernels"
bash ./compile_cuda.sh

results_dir=/jet/home/kchien/LLM-System-Final-Project-/benchmark_results
mkdir -p "${results_dir}"

python ./scripts/benchmark_fa2_gpu_vs_naive.py \
  --configs \
    bs4_seq512_embd512_heads8 \
    bs8_seq256_embd256_heads8 \
    bs8_seq256_embd512_heads8 \
    bs8_seq64_embd256_heads8 \
    bs2_seq1024_embd512_heads8 \
    bs1_seq2048_embd512_heads8 \
  --variants naive fa2 \
  --warmup 5 \
  --repeats 20 \
  --seed 2026 \
  --output-json "${results_dir}/fa2_gpu_bench_${SLURM_JOB_ID}.json"

echo "[INFO] End: $(date)"