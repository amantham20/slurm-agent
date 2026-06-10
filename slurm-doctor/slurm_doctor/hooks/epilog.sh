#!/usr/bin/env bash
# OPTIONAL per-node epilog for slurm-doctor (Epilog=.../epilog.sh).
#
# Runs on each compute node as root right after a job ends, while node-local
# state is still fresh. It snapshots dmesg/GPU state for failed-looking jobs
# into the shared report dir so collect.py can use it even if the node is
# later rebooted or drained. Keep it fast and silent - epilog failures DRAIN
# the node, so everything here is best-effort with || true.

SNAPDIR="${SLURM_DOCTOR_REPORT_DIR:-/data/jobs/.slurm-doctor}/node-snapshots"
JOB="${SLURM_JOB_ID:-unknown}"
NODE="$(hostname -s 2>/dev/null || echo node)"

mkdir -p "$SNAPDIR" 2>/dev/null || exit 0
OUT="$SNAPDIR/${JOB}.${NODE}.txt"

{
    echo "### slurm-doctor epilog snapshot job=$JOB node=$NODE $(date -Is)"
    echo "### dmesg (oom/segfault/gpu lines, last 100)"
    dmesg -T 2>/dev/null | grep -iE 'out of memory|oom|killed process|segfault|nvrm|xid' | tail -100
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "### nvidia-smi"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv 2>&1
    fi
} > "$OUT" 2>/dev/null || true

exit 0
