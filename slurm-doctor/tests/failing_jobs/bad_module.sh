#!/bin/bash
#SBATCH --job-name=sd_bad_module
#SBATCH --output=/data/sd_bad_module_%j.out
#SBATCH --error=/data/sd_bad_module_%j.err
#SBATCH --time=00:01:00
#SBATCH --mem=100M
# Batch shells don't source profile.d, so initialise Lmod explicitly first.
source /etc/profile.d/lmod.sh 2>/dev/null
module load nonexistent_module_xyz
