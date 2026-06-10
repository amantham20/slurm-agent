#!/bin/bash
#SBATCH --job-name=cuda_missing
#SBATCH --partition=cpu
#SBATCH --mem=50M
#SBATCH --time=00:02:00
#SBATCH --output=/data/jobs/%x-%j.out

# GPU workload submitted to a CPU-only node, without --gres=gpu.
# nvidia-smi is not installed on CPU nodes -> "command not found", exit 127.
nvidia-smi
