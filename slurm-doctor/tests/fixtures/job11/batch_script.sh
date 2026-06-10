#!/bin/bash
#SBATCH --job-name=mpi_bad_launcher
#SBATCH --partition=cpu
#SBATCH --mem=100M
#SBATCH --time=00:02:00
#SBATCH --ntasks=2
#SBATCH --output=/data/jobs/%x-%j.out

# Wrong launcher: mpirun outside srun, with an -n that does not match
# the allocation. On this cluster mpirun is not even installed.
mpirun -n 8 ./mpi_app
