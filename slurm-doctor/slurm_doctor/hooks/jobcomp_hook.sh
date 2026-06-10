#!/usr/bin/env bash
# slurm-doctor job-completion hook (JobCompType=jobcomp/script).
#
# slurmctld invokes this synchronously for EVERY finished job and blocks until
# it returns, so this script must come back in milliseconds: it only inspects
# the env vars SLURM passes (JOBID, JOBSTATE, ...) and, for failures, detaches
# a background `slurm-doctor suggest` with nohup. Never run `heal` from here -
# auto-resubmission is far too aggressive a site-wide default.
#
# Runs as SlurmUser. Useful env knobs:
#   SLURM_DOCTOR_BIN        path to the slurm-doctor entrypoint
#   SLURM_DOCTOR_HOOK_LOG   where background runs log (default below)
#   SLURM_DOCTOR_CACHE      artifact cache dir for the hook user

LOG="${SLURM_DOCTOR_HOOK_LOG:-/var/log/slurm/slurm-doctor-hook.log}"

case "${JOBSTATE:-}" in
    FAILED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE) ;;
    *) exit 0 ;;
esac
[ -n "${JOBID:-}" ] || exit 0

BIN="${SLURM_DOCTOR_BIN:-$(command -v slurm-doctor || echo /usr/local/bin/slurm-doctor)}"
[ -x "$BIN" ] || { echo "$(date -Is) jobcomp_hook: $BIN not executable" >> "$LOG" 2>/dev/null; exit 0; }

# Give the hook user a writable cache out of the box.
export SLURM_DOCTOR_CACHE="${SLURM_DOCTOR_CACHE:-/var/spool/slurm/slurm-doctor-cache}"

# Detach completely; slurmctld must not wait for the analysis.
nohup "$BIN" suggest "$JOBID" --from-hook >> "$LOG" 2>&1 &

exit 0
