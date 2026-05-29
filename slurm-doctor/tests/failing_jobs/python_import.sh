#!/bin/bash
#SBATCH --job-name=sd_python_import
#SBATCH --output=/data/sd_python_import_%j.out
#SBATCH --error=/data/sd_python_import_%j.err
#SBATCH --time=00:01:00
#SBATCH --mem=200M
python3 -c "import not_a_real_package_xyz"
