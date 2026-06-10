#!/bin/bash
#SBATCH --job-name=disk_full
#SBATCH --partition=cpu
#SBATCH --mem=50M
#SBATCH --time=00:02:00
#SBATCH --output=/data/jobs/%x-%j.out

# Fill a tiny private tmpfs until ENOSPC. Jobs on this cluster run as root
# in privileged containers, so the mount works; fall back to /dev/full
# (always returns ENOSPC on write) anywhere it does not.
mnt=$(mktemp -d)
if mount -t tmpfs -o size=8m tmpfs "$mnt" 2>/dev/null; then
    dd if=/dev/zero of="$mnt/fill" bs=1M count=64
    rc=$?
    rm -f "$mnt/fill"
    umount "$mnt"
else
    dd if=/dev/zero of=/dev/full bs=1M count=1
    rc=$?
fi
exit $rc
