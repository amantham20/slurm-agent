#!/bin/bash
#SBATCH --job-name=timeout
#SBATCH --partition=cpu
#SBATCH --mem=50M
#SBATCH --time=00:00:30
#SBATCH --output=/data/jobs/%x-%j.out

echo "starting long computation..."
sleep 600
echo "never reached"
