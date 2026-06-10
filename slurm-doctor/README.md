# slurm-doctor

An autonomous SLURM job failure analyst. It watches for failed jobs, gathers
every piece of evidence needed to explain them (accounting, submit script,
stdout/stderr, slurmd logs, dmesg, node and GPU state), diagnoses the root
cause with a declarative rule engine (optional Claude fallback), writes a
human + machine readable report, proposes concrete fixes as minimal diffs —
and, when it is safe, resubmits a patched version of the job.

```
$ slurm-doctor heal 2
Job 2: The job used more memory than it requested and slurmstepd killed it
(OverMemoryKill). Suggested fix: raise memory request to --mem=448M.
report: /data/jobs/.slurm-doctor/2/report.md
healed: resubmitted as job 14 with bump_memory
patched script: /data/jobs/oom.fix1.sh

$ sacct -j 14 -X --format=State,ReqMem     # a few minutes later
COMPLETED 448M
```

## How it works

```
            ┌──────────┐   ┌────────┐   ┌──────────┐   ┌─────────┐   ┌──────┐
 job id ──▶ │ collect  │──▶│ parse  │──▶│ diagnose │──▶│ report  │──▶│ fix  │
            └──────────┘   └────────┘   └──────────┘   └─────────┘   └──────┘
             sacct/sacct -B  state machine  Diagnosis +   report.md     patch +
             scontrol, logs, + rules/*.yaml ProposedFix   report.json   sbatch
             dmesg, GPU      (+ LLM opt-in)  (ranked)     (+ evidence)  (gated)
```

1. **Collect** (`collect.py`) — for a job id, pull the full accounting record
   (all steps), the submit script (`sacct -B`, falling back to
   `scontrol show job -dd` / `Command=`), capped stdout/stderr (4 MB head+tail
   by default), per-node `scontrol show node`, the job's window of
   slurmd/jobcomp logs, dmesg OOM/segfault/GPU lines, `nvidia-smi` when the
   job held GPUs, and the redacted stored environment. Everything is cached
   under `~/.cache/slurm-doctor/<jobid>/` — re-runs never re-hit sacct.
2. **Parse** (`parse.py`) — layer 1 maps State x ExitCode x Reason to a coarse
   class (it will not confuse `CANCELLED+0:0`, a user scancel, with
   `CANCELLED+0:15`, a signalled kill). Layer 2 runs declarative rules from
   `slurm_doctor/rules/*.yaml`; first match wins per category, multiple
   categories may fire, and every match must cite evidence (file + line).
3. **Diagnose** (`diagnose.py`) — combines hits into a `Diagnosis` with ranked
   `ProposedFix`es whose parameters are computed from the evidence
   (`bump_memory` = ceil(MaxRSS x 1.3) rounded up; `bump_time` =
   ceil(Elapsed x 1.5) capped at partition MaxTime; ...). If **no rule fires**
   and you passed `--llm` (or `SLURM_DOCTOR_LLM=1`), a compact, secret-redacted
   excerpt goes to Claude (`claude-sonnet-4-5`) with a strict JSON schema;
   LLM confidence is capped below the auto-heal gate by construction.
4. **Report** (`report.py`) — writes `report.md` (TL;DR, what ran, why it
   failed with quoted log lines, ranked fixes each with a unified diff,
   verification plan) and schema-versioned `report.json` to
   `/data/jobs/.slurm-doctor/<jobid>/`.
5. **Fix** (`fix.py`) — surgical, idempotent, marker-guarded patches written
   to `<workdir>/<original>.fix<N>.sh` (originals are never touched). A patch
   that would rewrite more than 20% of the script is downgraded to
   suggest-only.

## Commands

| command | effect |
|---|---|
| `slurm-doctor collect <jobid>` | fetch + cache evidence only |
| `slurm-doctor suggest <jobid>` | diagnose and write reports |
| `slurm-doctor patch <jobid>` | reports **and** patched script(s) |
| `slurm-doctor heal <jobid> [--yes]` | reports, patch, **resubmit** the top fix |
| `slurm-doctor sweep --since '1 hour ago'` | suggest for every recent failure |
| `slurm-doctor rules` | list loaded rules |

Global flags: `--cache-dir`, `--report-dir`, `--llm`, `-v/-vv`.

### heal safety model

`heal` resubmits **only** when all of these hold (any miss → it tells you to
rerun with `--yes`):

- top fix confidence ≥ 0.85 (`SLURM_DOCTOR_HEAL_CONFIDENCE`),
- fix kind in the allowlist — default `bump_memory,bump_time,add_set_eux`
  (`SLURM_DOCTOR_AUTOFIX_KINDS`),
- the kind is not in the always-confirm classes (MPI, GPU, constraints —
  those are easy to make worse than the original failure).

Every resubmission carries `--comment="slurm-doctor:fix=<kind>:parent=<jobid>"`
and is recorded in `<report-dir>/chain.json`; a job whose ancestry already
contains 2 heals is refused (fix-loop protection). If the original is still
pending requeue, the resubmission gets `--dependency=afternotok:<jobid>`.

## Fix kinds

| fix_kind | action | auto-heal? |
|---|---|---|
| `bump_memory` | `--mem` := ceil(MaxRSS x 1.3), rounded up | yes |
| `bump_time` | `--time` := ceil(Elapsed x 1.5), capped at partition MaxTime | yes |
| `add_set_eux` | insert `set -euo pipefail` after the header | yes |
| `prepend_modules` | insert Lmod init (+ `module load`s) above the first command | --yes |
| `fix_path` | re-point a missing path at an unambiguous WorkDir match | --yes |
| `swap_mpi_launcher` | `mpirun/mpiexec [-n N]` → `srun --mpi=pmix` | always --yes |
| `pin_gpu_visible` | stop fighting SLURM over `CUDA_VISIBLE_DEVICES` | always --yes |
| `request_constraint` | add `--constraint=` (+ partition/gres) | always --yes |
| `add_requeue_guard` | `--requeue` + `--open-mode=append` | --yes |

## Rules

25+ signatures ship in `slurm_doctor/rules/*.yaml`: missing module / module
unknown, missing executable, command not found, permission denied, MPI launch
failures (PMIx/Hydra/`srun: launch failed`/mpirun-not-found/slot mismatch),
CUDA missing & driver mismatch, GPU OOM, CPU OOM (state, OverMemoryKill log
and kernel oom-killer variants), walltime, deadline, disk quota / disk full,
stale network mounts, Python `ModuleNotFoundError`, segfault (stderr and
signal-11 variants), FlexLM/ANSYS license failures, node fail / drained
mid-job, boot fail, preemption, cancelled-by-user/admin/signal.

A rule is a few lines of YAML:

```yaml
- id: missing_module
  category: environment
  match:
    stderr_regex: '(module: command not found|Lmod has detected the following error)'
  hint: "Module system is not loaded or the module name is wrong."
  fix_kind: prepend_modules
  confidence: 0.9
```

Match keys: `stderr_regex`, `stdout_regex`, `script_regex`, `slurmd_regex`,
`dmesg_regex`, `sacct_regex`, `jobcomp_regex` (all AND-ed), plus `state`,
`exit_signal`, `min_exit_code`, `cancelled_by`, `reason_regex`. Named groups
(e.g. `(?P<missing_path>...)`) become fix parameters. Drop extra `*.yaml`
files into the rules directory to extend; PyYAML is optional (a built-in
mini-parser covers the shipped subset, with parity enforced by tests).

## Integration modes

### A. Manual CLI

`make doctor-install` installs the package into the `slurmctld` container
(plus bash completion). Use the commands above ad hoc.

### B. Job-completion hook (recommended)

```
make doctor-install-hook
```

appends to the **live** `/etc/slurm/slurm.conf` (exact lines in
`docker/slurm.conf.snippet`):

```
JobCompType=jobcomp/script
JobCompLoc=/opt/slurm-doctor/hooks/jobcomp_hook.sh
```

then runs `scontrol reconfigure`. The hook returns in milliseconds (it only
inspects env vars and detaches `nohup slurm-doctor suggest $JOBID`), only
reacts to FAILED/TIMEOUT/OUT_OF_MEMORY/NODE_FAIL/BOOT_FAIL/DEADLINE, never
heals, and logs TL;DRs to `/var/log/slurm/slurm-doctor-hook.log`.
`make doctor-uninstall-hook` restores file-based job completion. slurm-doctor
itself never touches scheduler config outside this explicit target.

An optional per-node `Epilog` (`hooks/epilog.sh`) snapshots dmesg/nvidia-smi
at job end for evidence that survives node reboots; enable it by uncommenting
the line in the snippet.

### C. Periodic sweep

`slurm-doctor sweep --since '1 hour ago'` lists recent failures (preferring
`slurmrestd` — unix socket or `:6820` with a `scontrol token` JWT — and
falling back to `sacct -X --state=...` cleanly) and writes reports for
anything not already reported. Ship-with units:

- `systemd/slurm-doctor-sweep.{service,timer}` — every 15 min on a real host
- `systemd/crontab.example` — cron flavour
- `docker/slurm-doctor.compose.yaml` — sweep sidecar overlay for this cluster

## Testing

```
make test-doctor      # unit tests + live end-to-end inside slurmctld
```

- Unit tests (`pytest`, no cluster needed) run against captured fixtures in
  `tests/fixtures/` — real bundles from real failures of every category.
- `tests/test_end_to_end.py` (marker `e2e`) submits each script in
  `tests/failing_jobs/` (oom, timeout, bad_module, missing_exec,
  mpi_bad_launcher, segfault, cuda_missing, python_import, disk_full), waits,
  runs `suggest`, and asserts category + fix kind; it also heals the OOM job
  and asserts the resubmission COMPLETEs. The rule engine must resolve every
  case with `used_llm == false`.

## Configuration

| env var | default | meaning |
|---|---|---|
| `SLURM_DOCTOR_CACHE` | `~/.cache/slurm-doctor` | raw artifact cache |
| `SLURM_DOCTOR_REPORT_DIR` | `/data/jobs/.slurm-doctor` | reports + chain ledger |
| `SLURM_DOCTOR_MAX_IO_BYTES` | `4194304` | per stdout/stderr cap (head+tail) |
| `SLURM_DOCTOR_CMD_TIMEOUT` | `30` | seconds per external command |
| `SLURM_DOCTOR_LLM` | off | enable the Claude fallback |
| `SLURM_DOCTOR_LLM_MODEL` | `claude-sonnet-4-5` | fallback model |
| `SLURM_DOCTOR_AUTOFIX_KINDS` | `bump_memory,bump_time,add_set_eux` | heal allowlist |
| `SLURM_DOCTOR_HEAL_CONFIDENCE` | `0.85` | auto-heal gate |
| `SLURM_DOCTOR_MAX_HEAL_CHAIN` | `2` | heal-loop refusal depth |
| `SLURM_DOCTOR_RESTD` | autodetect | slurmrestd endpoint (`unix://...` or `http://...`) |
| `SLURM_DOCTOR_NODE_EXEC` | local | e.g. `ssh {node}` to run node commands remotely |
| `SLURM_DOCTOR_SLURMD_LOG` | `/var/log/slurm/slurmd.log` | slurmd log location |

All shell-outs go through one wrapper: logged, captured, timed out, and never
`shell=True`. The only state is the cache dir and the report dir; every
command is safe to re-run.

## Cluster notes (this repo)

The vendored cluster config enables `JobAcctGatherType=jobacct_gather/linux`
with `OverMemoryKill` and 5s sampling, plus
`AccountingStoreFlags=job_script,job_comment,job_env`. Without those, sacct
has no `MaxRSS` (so `bump_memory` cannot size anything), over-memory jobs are
never killed under `proctrack/linuxproc`, and submit scripts/comments are
lost once a job leaves slurmctld memory. On your own site, check
`scontrol show config | grep -i jobacctgather` before expecting OOM
diagnoses to carry measured peaks.
