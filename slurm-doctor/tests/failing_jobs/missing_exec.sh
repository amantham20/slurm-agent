#!/bin/bash
#SBATCH --job-name=sd_missing_exec
#SBATCH --output=/data/sd_missing_exec_%j.out
#SBATCH --error=/data/sd_missing_exec_%j.err
#SBATCH --time=00:01:00
#SBATCH --mem=100M
echo "about to run a binary that does not exist on $(hostname)"
./binary_that_doesnt_exist --do-science
