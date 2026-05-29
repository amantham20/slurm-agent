#!/bin/bash
#SBATCH --job-name=sd_timeout
#SBATCH --output=/data/sd_timeout_%j.out
#SBATCH --error=/data/sd_timeout_%j.err
#SBATCH --time=00:00:30
#SBATCH --mem=100M
# Sleeps well past the 30s wall limit so SLURM kills it -> State=TIMEOUT.
echo "sleeping 600s under a 30s wall limit on $(hostname)"
sleep 600
