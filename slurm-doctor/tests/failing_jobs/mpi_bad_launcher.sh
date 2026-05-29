#!/bin/bash
#SBATCH --job-name=sd_mpi_bad
#SBATCH --output=/data/sd_mpi_bad_%j.out
#SBATCH --error=/data/sd_mpi_bad_%j.err
#SBATCH --time=00:01:00
#SBATCH --ntasks=2
#SBATCH --mem=200M
# The classic anti-pattern: launch MPI badly. If a real mpirun exists we use it
# with a task count that mismatches the allocation (-np 4 under 2 tasks). On a
# cluster without an MPI runtime (like this one) we instead ask srun for an MPI
# plugin the site doesn't provide (pmix) — both are genuine launcher failures.
if command -v mpirun >/dev/null 2>&1; then
    mpirun -np 4 hostname
else
    srun --mpi=pmix -n 4 hostname
fi
