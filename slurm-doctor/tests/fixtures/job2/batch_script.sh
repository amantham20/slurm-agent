#!/bin/bash
#SBATCH --job-name=oom
#SBATCH --partition=cpu
#SBATCH --mem=50M
#SBATCH --time=00:05:00
#SBATCH --output=/data/jobs/%x-%j.out

# Allocate ~300MB (6x the request) and hold it so the per-task RSS poll
# (JobAcctGatherFrequency=task=5 + OverMemoryKill) catches and kills us.
python3 - <<'PY'
import time
data = bytearray(300 * 1024 * 1024)
for i in range(0, len(data), 4096):
    data[i] = 1
print("allocated 300MB, holding...", flush=True)
time.sleep(180)
PY
