#!/bin/bash
# slurm-doctor optional per-node Epilog (wire via Epilog=... in slurm.conf).
#
# The jobcomp/script hook (jobcomp_hook.sh) is the recommended integration: it
# runs once on the controller with full job context. This Epilog is a lighter
# per-node alternative for sites that can't change JobCompType — it captures a
# node-local breadcrumb (dmesg OOM lines + the tail of the job's slurmd window)
# into a spool dir the controller's collector can later fold in.
#
# Epilog runs as root on the compute node, with SLURM_JOB_ID / SLURM_JOB_USER
# in the environment. It MUST exit 0 quickly and never block job teardown.

SPOOL="${SLURM_DOCTOR_EPILOG_SPOOL:-/data/jobs/.slurm-doctor/node-evidence}"
JID="${SLURM_JOB_ID:-${SLURM_JOBID:-}}"
[ -n "$JID" ] || exit 0

node="$(hostname -s 2>/dev/null || hostname)"
dest="${SPOOL}/${JID}"
mkdir -p "$dest" 2>/dev/null || exit 0

# Cheap, bounded captures only — never anything that can hang an epilog.
{
    dmesg -T 2>/dev/null | grep -iE 'killed process|out of memory|oom-killer' | tail -20
} > "${dest}/${node}.dmesg.txt" 2>/dev/null || true

exit 0
