#!/bin/bash
#SBATCH --job-name=sd_cuda_missing
#SBATCH --output=/data/sd_cuda_missing_%j.out
#SBATCH --error=/data/sd_cuda_missing_%j.err
#SBATCH --time=00:01:00
#SBATCH --mem=200M
# Job assumes a GPU but did NOT request one (no --gres=gpu) so it lands on a
# CPU node. With NVIDIA tooling present this query fails against the driver; on
# an image without nvidia-smi we emit the byte-identical error a real CPU-only
# node produces, so the failure mode is faithful and deterministic.
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || exit 1
else
    echo "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running." >&2
    exit 1
fi
