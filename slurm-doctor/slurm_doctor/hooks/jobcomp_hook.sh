#!/bin/bash
# slurm-doctor JobComp hook — wired via SLURM's JobCompType=jobcomp/script.
#
# SLURM runs this script (as SlurmUser) on EVERY job completion and passes the
# job's data in the environment: JOBID, JOBSTATE, EXITCODE, WORK_DIR, NODES,
# STDOUT/STDERR, LIMIT, etc. On a failure state we background
# `slurm-doctor suggest <JOBID>` and return immediately.
#
# We NEVER `heal` from here — automatic resubmission is far too aggressive a
# default for a site-wide completion hook. Healing stays a deliberate human/CLI
# action.
#
# Hard requirement: return within ~50ms. slurmctld blocks on this script, so
# all real work is detached with setsid and fully redirected; the foreground
# path does nothing but a state check and a fork.

# --- configuration (override via Environment in the systemd unit, or by
#     exporting before slurmctld starts) --------------------------------------
SD_HOME="${SLURM_DOCTOR_HOME:-/opt/slurm-doctor}"
SD_PYTHON="${SLURM_DOCTOR_PYTHON:-python3}"
SD_REPORTS="${SLURM_DOCTOR_REPORTS:-/data/jobs/.slurm-doctor}"
SD_CACHE="${SLURM_DOCTOR_CACHE:-/data/jobs/.slurm-doctor/.cache}"
SD_LOG="${SLURM_DOCTOR_HOOK_LOG:-/data/jobs/.slurm-doctor/hook.log}"

# --- only act on failure states --------------------------------------------
# (jobcomp/script sets JOBSTATE to the bare state string, e.g. FAILED)
case "${JOBSTATE:-}" in
    FAILED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE) ;;
    *) exit 0 ;;
esac
[ -n "${JOBID:-}" ] || exit 0

# --- detach the real work; return immediately ------------------------------
# setsid gives the worker its own session so it survives this script exiting;
# stdin/stdout/stderr are fully redirected so slurmctld never blocks on a pipe.
setsid "${BASH:-/bin/bash}" -c "
    cd '${SD_HOME}' 2>/dev/null || exit 0
    echo \"[\$(date -u +%FT%TZ)] hook: job ${JOBID} state=${JOBSTATE} exit=${EXITCODE:-?} -> suggest\" >> '${SD_LOG}' 2>&1
    '${SD_PYTHON}' -m slurm_doctor.cli --hook \
        --cache-dir '${SD_CACHE}' --reports-dir '${SD_REPORTS}' \
        suggest '${JOBID}' >> '${SD_LOG}' 2>&1
" >/dev/null 2>&1 </dev/null &

exit 0
