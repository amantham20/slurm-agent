#!/bin/bash
#SBATCH --job-name=oom
#SBATCH --output=/data/oom_%j.out
#SBATCH --error=/data/oom_%j.err
#SBATCH --mem=200M
#SBATCH --time=00:01:00

# This cluster runs with cgroup memory enforcement OFF
# (cgroup.conf: ConstrainRAMSpace=no), so SLURM does NOT kill jobs that
# exceed --mem. We approximate enforcement with a ulimit derived from the
# scheduler's requested memory so the test fails deterministically here
# AND so the fix (bumping --mem) actually relieves the failure on a re-run.
ulimit -v $(( ${SLURM_MEM_PER_NODE:-100} * 1024 ))

python3 -u -c '
import time
data = []
# Allocate 20 MB / second so jobacct_gather (poll=5s) samples MaxRSS before
# the failure, and the rule engine sees MaxRSS/ReqMem >= 0.9.
for i in range(20):
    data.append(bytearray(20 * 1024 * 1024))
    time.sleep(1)
    print(f"allocated {(i+1)*20} MB", flush=True)
'
