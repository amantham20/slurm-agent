#!/bin/bash
#SBATCH --job-name=missing_exec
#SBATCH --partition=cpu
#SBATCH --mem=50M
#SBATCH --time=00:02:00
#SBATCH --output=/data/jobs/%x-%j.out

./binary_that_doesnt_exist --input data.csv
