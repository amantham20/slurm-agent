# slurm-doctor

An autonomous diagnostician for failed SLURM jobs. It watches for failures,
figures out *why* a job died (citing the actual log lines as evidence), writes
a structured report, proposes concrete fixes, and — only when it's safe —
resubmits a patched version of the job.

It runs three ways, your choice per site:

| Mode | How | When to use |
|------|-----|-------------|
| **A — Manual CLI** | `slurm-doctor suggest/patch/heal <jobid>` | interactive debugging |
| **B — Completion hook** | `JobCompType=jobcomp/script` → auto-`suggest` on every failure | hands-off, site-wide |
| **C — Periodic sweep** | `slurm-doctor sweep --since '1 hour ago'` via systemd timer or cron | catch-up / batch |

---

## Pipeline

```
collect  ─►  parse  ─►  diagnose  ─►  report        (suggest)
                                  └─►  fix ─► sbatch  (patch / heal)
```

1. **collect** (`collect.py`) — given a job id, gathers everything needed to
   diagnose *without re-running*: `sacct` accounting (all steps), the submit
   script (`sacct -B`, falling back to `scontrol Command=`, then to a
   `WorkDir`/`JobName` heuristic for purged jobs), stdout/stderr (head+tail
   truncated above 4 MB), per-node `scontrol show node`, `slurmd.log` sliced to
   the job's window, `dmesg` OOM lines (only when warranted), and GPU state
   (only when `AllocTRES` has `gres/gpu`). Everything is cached under
   `~/.cache/slurm-doctor/<jobid>/` so re-runs never re-hit `sacct`.

2. **parse** (`parse.py`) — a layered classifier:
   - *Layer 1* — a SLURM state machine over `State × ExitCode × Reason`
     (`TIMEOUT`, `OUT_OF_MEMORY`, `NODE_FAIL`, `BOOT_FAIL`, `PREEMPTED`,
     `DEADLINE`, `CANCELLED` split into user-vs-signalled, …).
   - *Layer 2* — declarative pattern rules from `rules/*.yaml`.
   - *Layer 3* — an optional LLM fallback (opt-in only; see below).

3. **diagnose** (`diagnose.py`) — synthesises a `Diagnosis`: root cause,
   contributing factors, confidence, and a ranked list of `ProposedFix`es.

4. **report** (`report.py`) — writes `report.md` (human) and `report.json`
   (machine, `schema_version: 1`) to `/data/jobs/.slurm-doctor/<jobid>/`. Every
   diagnosis quotes the specific evidence line it's based on.

5. **fix** (`fix.py`) — turns a `ProposedFix` into a minimal, idempotent patch
   of the submit script, written to `<workdir>/<original>.fix<N>.sh` (never
   overwriting the original).

---

## CLI

```bash
slurm-doctor suggest <jobid>   # report only — never touches anything else
slurm-doctor patch   <jobid>   # + write patched script(s) for safe fixes
slurm-doctor heal    <jobid>   # + sbatch the top safe fix (gated; see below)
slurm-doctor sweep --since '1 hour ago'   # diagnose every recent failure
```

Useful flags: `--llm` / `SLURM_DOCTOR_LLM=1` (enable Layer 3), `--yes` (override
the heal gate), `--dry-run` (heal without submitting), `--reports-dir`,
`--cache-dir`, `--rules-dir`, `--allow` (override the auto-fix allowlist).

### The heal gate

`heal` only auto-applies a fix when **confidence ≥ 0.85 AND the fix kind is in
the allowlist** (default `bump_memory`, `bump_time`, `add_set_eux`). Anything
touching MPI, GPUs, or constraints needs `--yes` — those are easy to make worse
than the original failure. Every resubmission carries
`--comment=slurm-doctor:fix=<kind>:parent=<jobid>`, and slurm-doctor refuses to
heal a job whose ancestry has already been healed twice (fix-loop guard).

---

## Fix kinds

| fix_kind | what it does | auto? |
|----------|--------------|:----:|
| `bump_memory` | `--mem` → `ceil(MaxRSS × 1.3)` rounded up to GB | ✅ |
| `bump_time` | `--time` → `ceil(Elapsed × 1.5)`, capped at partition MaxTime | ✅ |
| `add_set_eux` | insert `set -euo pipefail` | ✅ |
| `prepend_modules` | insert an `lmod` init + `module load` block | needs `--yes` |
| `fix_path` | rewrite a missing path to a close match in WorkDir | needs `--yes` |
| `swap_mpi_launcher` | `mpirun`/`mpiexec` → `srun --mpi=pmix` | needs `--yes` |
| `request_constraint` | add `--constraint=<feature>` | needs `--yes` |
| `add_requeue_guard` | add `--requeue` + a restart backoff | needs `--yes` |
| `pin_gpu_visible` | pin `CUDA_VISIBLE_DEVICES` from the SLURM allocation | needs `--yes` |

Patches are surgical: a directive is rewritten in place, or one block is
inserted after the `#SBATCH` header. A patch that would rewrite more than ~20%
of the original lines is automatically downgraded to *suggest*.

---

## Rules

Rules are declarative YAML (`slurm_doctor/rules/*.yaml`):

```yaml
- id: missing_module
  title: Tried to "module load" a module that isn't available
  category: modules
  match:
    stderr_regex: '(module: command not found|Lmod has detected the following error)'
  hint: "Module system isn't loaded or the module name is wrong."
  fix_kind: prepend_modules
  confidence: 0.90
```

Supported `match:` predicates (validated at load time — a typo fails fast):
`stderr_regex`, `stdout_regex`, `slurmd_log_regex`, `dmesg_regex`,
`exit_code_in`, `reason_regex`, `state_in` (Layer-1 categories), and
`max_rss_ratio_above`. Rules are ordered; first match per category wins.

The shipped pack covers all the families the project asks for: missing
module/executable, permission denied, MPI launch failure, CUDA driver missing,
GPU OOM, CPU/kernel OOM, SLURM OOM state, memory-pressure heuristic, walltime,
disk full/quota, stale network mount, Python `ModuleNotFoundError`,
Python `MemoryError`, segfault, license-server unreachable, node drained.

---

## Install & integration

```bash
make install-hook     # install into slurmctld + wire JobCompType=jobcomp/script
make uninstall-hook   # restore jobcomp/filetxt
make test-doctor      # end-to-end tests against the live cluster
make test-doctor-unit # host unit tests (no cluster needed)
```

`make install-hook` copies the package to `/opt/slurm-doctor`, ensures the
report dirs are writable by `SlurmUser`, appends the two lines from
`docker/slurm.conf.snippet`, and runs `scontrol reconfigure` (no restart
needed — `JobCompType` reloads on 25.11). The hook (`hooks/jobcomp_hook.sh`)
runs on every completion, acts only on failure states, and backgrounds
`slurm-doctor suggest` so it returns in single-digit milliseconds. It never
heals — automatic resubmission stays a deliberate CLI action.

**Mode C** ships `docker/slurm-doctor-sweep.{service,timer}` and
`docker/slurm-doctor.cron`; sweep is read-only (`suggest` only) and idempotent,
skipping jobs that already have a report.

---

## Safety / quality bar

- No external state beyond the cache dir and the report dir; re-running is safe
  and idempotent.
- Every shell-out goes through one wrapper that logs the command, enforces a
  timeout, captures stdout/stderr, and **never** uses `shell=True` with
  interpolation.
- The LLM is **opt-in** (`--llm` / `SLURM_DOCTOR_LLM=1`), never default; the
  rule engine resolves 100% of the bundled failing-job tests with zero LLM
  calls. Secret-shaped values (`AWS_`, `*_TOKEN`, `*_KEY`, `*PASSWORD*`) are
  never sent to the LLM.
- No diagnosis without a citation back to a specific log line.
- slurm-doctor never edits `slurm.conf` on its own — only the explicit
  `install-hook` target does, and it backs the file up first.

---

## Tests

```bash
make test-doctor-unit   # ~90 host unit tests, no SLURM needed
make test-doctor        # 9 deliberately-broken jobs, live cluster
```

`tests/failing_jobs/` has one script per failure category; `test_end_to_end.py`
submits each, waits, runs `suggest`, and asserts the category and a proposed fix
kind. Unit tests mock `sacct`/`scontrol` against captured fixtures in
`tests/fixtures/`.
