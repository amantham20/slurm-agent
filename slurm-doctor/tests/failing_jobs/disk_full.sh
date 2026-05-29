#!/bin/bash
#SBATCH --job-name=sd_disk_full
#SBATCH --output=/data/sd_disk_full_%j.out
#SBATCH --error=/data/sd_disk_full_%j.err
#SBATCH --time=00:01:00
#SBATCH --mem=200M
# Fill a small tmpfs to force a real "No space left on device" from dd.
mnt="/tmp/sd_tmpfs_$$"
mkdir -p "$mnt"
mount -t tmpfs -o size=8M tmpfs "$mnt" || { echo "could not mount tmpfs" >&2; exit 1; }
dd if=/dev/zero of="$mnt/fill" bs=1M count=64
rc=$?
umount "$mnt" 2>/dev/null; rmdir "$mnt" 2>/dev/null
exit $rc
