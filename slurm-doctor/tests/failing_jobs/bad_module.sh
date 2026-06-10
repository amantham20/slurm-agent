#!/bin/bash
#SBATCH --job-name=bad_module
#SBATCH --partition=cpu
#SBATCH --mem=50M
#SBATCH --time=00:02:00
#SBATCH --output=/data/jobs/%x-%j.out

# Batch shells don't source /etc/profile.d, so the `module` shell function
# is not defined here -> "module: command not found" and exit 127.
module load nonexistent_module_xyz
