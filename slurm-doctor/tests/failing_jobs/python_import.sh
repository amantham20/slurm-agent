#!/bin/bash
#SBATCH --job-name=python_import
#SBATCH --partition=cpu
#SBATCH --mem=50M
#SBATCH --time=00:02:00
#SBATCH --output=/data/jobs/%x-%j.out

python3 -c "import not_a_real_package_xyz"
