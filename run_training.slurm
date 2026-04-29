#!/bin/bash
#SBATCH --job-name=train-hw3
#SBATCH -p GPU-shared
#SBATCH --gpus=v100:1
#SBATCH -t 04:00:00
#SBATCH -o /jet/home/kchien/llmsys_hw3/training_slurm_%j.log
#SBATCH --mem=20G

cd /jet/home/kchien/llmsys_hw3

# Load CUDA module
module load cuda

# Activate Python environment
source .venv/bin/activate

# Run training (short run for sanity check)
echo "Starting training at $(date)"
python project/run_machine_translation.py \
	--learning_rate=0.02 \
	--n_epochs=4 \
	--grad_clip=1.0 \
	2>&1 | tee training_output_${SLURM_JOB_ID}.log
echo "Training completed at $(date) with exit code $?"
