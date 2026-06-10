"""Phase 1 — Collect.

Given a job id, gather everything needed to diagnose the failure without
re-running the job, and cache it as raw artifacts under
``<cache_dir>/<jobid>/`` so a re-run never re-hits sacct.

Artifacts written (all optional except sacct):

========================  ====================================================
sacct.parsable2.txt       full accounting record, all steps
batch_script.sh           submit script (sacct -B, scontrol -dd, or Command=)
scontrol_job.txt          scontrol show job -dd (absent if job purged)
scontrol_partition.txt    scontrol show partition <partition>
stdout.txt / stderr.txt   job I/O, head+tail capped
nodes/<node>.txt          scontrol show node, per node
slurmd.filtered.log       slurmd/slurmstepd lines mentioning the job
jobcomp.filtered.log      jobcomp/filetxt record for the job, if present
dmesg.txt                 OOM-killer / GPU / segfault lines from dmesg -T
gpu.txt                   nvidia-smi snapshot (only if job allocated GPUs)
env.redacted.txt          stored job environment (sacct --env-vars), redacted
manifest.json             index + parsed summary of all of the above
========================  ====================================================
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

from . import __version__
from .config import Config
from .util import (
    CmdResult,
    expand_filename_pattern,
    head_tail_cap,
    parse_slurm_timestamp,
    read_file_capped,
    redact,
    run,
)

log = logging.getLogger("slurm_doctor.collect")

MANIFEST_SCHEMA = 1

SACCT_FORMAT = (
    "JobID,JobIDRaw,JobName,User,Partition,State,ExitCode,DerivedExitCode,"
    "Reason,Submit,Start,End,Elapsed,Timelimit,ReqMem,MaxRSS,MaxVMSize,"
    "AveCPU,ReqCPUS,AllocCPUS,ReqTRES,AllocTRES,NodeList,NNodes,WorkDir,"
    "Comment"
)

TERMINAL_STATES = (
    "COMPLETED",
    "FAILED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "BOOT_FAIL",
    "DEADLINE",
    "CANCELLED",
    "PREEMPTED",
    "REVOKED",
)

SLURMD_LOG = os.environ.get("SLURM_DOCTOR_SLURMD_LOG", "/var/log/slurm/slurmd.log")
JOBCOMP_LOG = os.environ.get("SLURM_DOCTOR_JOBCOMP_LOG", "/var/log/slurm/jobcomp.log")


# --------------------------------------------------------------------------
# Bundle: a typed view over the cache directory
# --------------------------------------------------------------------------


@dataclasses.dataclass
class Bundle:
    jobid: str
    dir: Path
    manifest: dict

    # ---- raw artifact accessors -------------------------------------------------
    def _read(self, name: str) -> str | None:
        p = self.dir / name
        try:
            return p.read_text(errors="replace")
        except OSError:
            return None

    @property
    def sacct_raw(self) -> str | None:
        return self._read("sacct.parsable2.txt")

    @property
    def script(self) -> str | None:
        return self._read("batch_script.sh")

    @property
    def scontrol_job(self) -> str | None:
        return self._read("scontrol_job.txt")

    @property
    def stdout_text(self) -> str | None:
        return self._read("stdout.txt")

    @property
    def stderr_text(self) -> str | None:
        return self._read("stderr.txt")

    @property
    def slurmd_log(self) -> str | None:
        return self._read("slurmd.filtered.log")

    @property
    def jobcomp_log(self) -> str | None:
        return self._read("jobcomp.filtered.log")

    @property
    def dmesg(self) -> str | None:
        return self._read("dmesg.txt")

    @property
    def gpu(self) -> str | None:
        return self._read("gpu.txt")

    @property
    def partition_info(self) -> str | None:
        return self._read("scontrol_partition.txt")

    def node_info(self, node: str) -> str | None:
        return self._read(f"nodes/{node}.txt")

    # ---- parsed views -----------------------------------------------------------
    @property
    def records(self) -> list[dict]:
        return parse_parsable2(self.sacct_raw or "")

    @property
    def parent(self) -> dict:
        for rec in self.records:
            if "." not in rec.get("JobID", "."):
                return rec
        return self.records[0] if self.records else {}

    @property
    def steps(self) -> list[dict]:
        return [r for r in self.records if "." in r.get("JobID", "")]

    @property
    def state(self) -> str:
        return (self.parent.get("State") or "").strip()

    @property
    def is_terminal(self) -> bool:
        st = self.state.split()[0] if self.state else ""
        return any(st.startswith(t) for t in TERMINAL_STATES)

    @property
    def nodes(self) -> list[str]:
        return self.manifest.get("nodes", [])

    @property
    def workdir(self) -> str | None:
        return self.parent.get("WorkDir") or None

    def max_rss_bytes(self) -> int | None:
        from .util import parse_mem_to_bytes

        best = None
        for rec in self.records:
            b = parse_mem_to_bytes(rec.get("MaxRSS") or "")
            if b is not None and (best is None or b > best):
                best = b
        return best

    def scontrol_field(self, key: str) -> str | None:
        kv = parse_scontrol_kv(self.scontrol_job or "")
        return kv.get(key)

    def evidence_sources(self) -> dict[str, str]:
        """Named text sources the rule engine can match against."""
        out: dict[str, str] = {}
        for name, attr in (
            ("stderr", "stderr_text"),
            ("stdout", "stdout_text"),
            ("script", "script"),
            ("slurmd", "slurmd_log"),
            ("dmesg", "dmesg"),
            ("sacct", "sacct_raw"),
            ("jobcomp", "jobcomp_log"),
        ):
            v = getattr(self, attr)
            if v:
                out[name] = v
        return out


# --------------------------------------------------------------------------
# Output parsers
# --------------------------------------------------------------------------


def parse_parsable2(text: str) -> list[dict]:
    """Parse `sacct --parsable2` output (header row + | separated rows)."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    header = lines[0].split("|")
    out = []
    for ln in lines[1:]:
        cols = ln.split("|")
        if len(cols) < 2:
            continue
        rec = dict(zip(header, cols))
        out.append(rec)
    return out


_SC_KEY = re.compile(r"(?:(?<=\s)|^)([A-Za-z][A-Za-z0-9_/:]*)=")


def parse_scontrol_kv(text: str) -> dict[str, str]:
    """Parse `scontrol show ...` key=value output.

    Handles whole-line values (Command=, WorkDir=, StdOut= live alone on
    their line) as well as multiple key=value pairs per line.
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        keys = list(_SC_KEY.finditer(line))
        for i, m in enumerate(keys):
            start = m.end()
            end = keys[i + 1].start() if i + 1 < len(keys) else len(line)
            val = line[start:end].strip()
            out.setdefault(m.group(1), val)
    return out


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------


def _node_argv(cfg: Config, node: str, argv: list[str]) -> list[str]:
    """Build the argv to run *argv* "on" *node*.

    By default commands run locally: in this docker cluster the controller
    shares /var/log/slurm and the kernel with the workers, so local reads see
    node-level state.  Set SLURM_DOCTOR_NODE_EXEC='ssh {node}' (or
    'docker exec {node}') on real clusters.
    """
    if not cfg.node_exec:
        return argv
    prefix = [p.format(node=node) for p in cfg.node_exec.split()]
    return prefix + argv


def wait_until_accounted(jobid: str, cfg: Config, max_wait: float = 90.0) -> bool:
    """Poll sacct until the job shows a terminal state with an End time.

    Used by the jobcomp hook: at the time the hook fires, slurmdbd may not
    have flushed the final record yet.
    """
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        r = run(
            ["sacct", "-j", str(jobid), "-X", "--parsable2", "--noheader",
             "--format=State,End"],
            timeout=cfg.cmd_timeout,
        )
        if r.ok and r.stdout.strip():
            state, end = (r.stdout.strip().splitlines()[0].split("|") + [""])[:2]
            st = state.split()[0] if state else ""
            if end not in ("", "Unknown") and any(
                st.startswith(t) for t in TERMINAL_STATES
            ):
                return True
        time.sleep(3)
    return False


def collect(jobid: str, cfg: Config, refresh: bool = False) -> Bundle:
    """Collect (or load from cache) the full evidence bundle for *jobid*."""
    jobid = str(jobid).strip()
    if not re.fullmatch(r"\d+(_\d+)?(\.\w+)?", jobid):
        raise ValueError(f"that does not look like a job id: {jobid!r}")
    cache = cfg.job_cache(jobid)
    manifest_path = cache / "manifest.json"

    if manifest_path.exists() and not refresh:
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            manifest = None
        if manifest and manifest.get("schema_version") == MANIFEST_SCHEMA:
            bundle = Bundle(jobid, cache, manifest)
            if bundle.is_terminal:
                log.debug("cache hit for job %s at %s", jobid, cache)
                return bundle
            log.info("job %s was not terminal when cached; re-collecting", jobid)

    cache.mkdir(parents=True, exist_ok=True)
    (cache / "nodes").mkdir(exist_ok=True)
    manifest: dict = {
        "schema_version": MANIFEST_SCHEMA,
        "slurm_doctor_version": __version__,
        "jobid": jobid,
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "artifacts": {},
        "commands": [],
    }

    def record_cmd(r: CmdResult) -> CmdResult:
        manifest["commands"].append(
            {"argv": r.argv, "rc": r.returncode, "duration": round(r.duration, 3)}
        )
        return r

    def save(name: str, text: str | None, meta: dict | None = None) -> None:
        if text is None:
            return
        p = cache / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        manifest["artifacts"][name] = {"bytes": len(text.encode())} | (meta or {})

    # 1. accounting record, all steps ------------------------------------------
    r = record_cmd(
        run(
            ["sacct", "-j", jobid, "--parsable2", f"--format={SACCT_FORMAT}"],
            timeout=cfg.cmd_timeout,
        )
    )
    if not r.ok or len(r.stdout.strip().splitlines()) < 2:
        raise RuntimeError(
            f"sacct returned no accounting data for job {jobid}: "
            f"{(r.stderr or r.stdout).strip() or 'empty output'}"
        )
    save("sacct.parsable2.txt", r.stdout)
    records = parse_parsable2(r.stdout)
    parent = next((x for x in records if "." not in x.get("JobID", ".")), records[0])

    # 2. scontrol view (may be gone if the job left slurmctld memory) ----------
    r = record_cmd(run(["scontrol", "show", "job", jobid, "-dd"], timeout=cfg.cmd_timeout))
    sc: dict[str, str] = {}
    if r.ok and r.stdout.strip():
        save("scontrol_job.txt", r.stdout)
        sc = parse_scontrol_kv(r.stdout)

    # 3. submit script ----------------------------------------------------------
    script = _collect_script(jobid, sc, cfg, record_cmd)
    save("batch_script.sh", script)

    # 4. stdout / stderr --------------------------------------------------------
    meta = {
        "jobid": jobid,
        "jobname": parent.get("JobName", ""),
        "user": parent.get("User", ""),
        "node": (parent.get("NodeList") or "").split(",")[0],
    }
    stdout_path, stderr_path = _io_paths(sc, script, parent, meta)
    for name, path in (("stdout.txt", stdout_path), ("stderr.txt", stderr_path)):
        if not path:
            continue
        text, fmeta = read_file_capped(path, cfg.max_io_bytes)
        save(name, text, fmeta)
        if text is None:
            manifest["artifacts"][name] = {"missing": True} | fmeta
    manifest["stdout_path"] = stdout_path
    manifest["stderr_path"] = stderr_path

    # 5. nodes ------------------------------------------------------------------
    nodes = _expand_nodelist(parent.get("NodeList") or "", cfg, record_cmd)
    manifest["nodes"] = nodes
    for node in nodes:
        r = record_cmd(run(["scontrol", "show", "node", node], timeout=cfg.cmd_timeout))
        if r.ok:
            save(f"nodes/{node}.txt", r.stdout)

    # partition limits (needed by bump_time)
    if parent.get("Partition"):
        r = record_cmd(
            run(["scontrol", "show", "partition", parent["Partition"]], timeout=cfg.cmd_timeout)
        )
        if r.ok:
            save("scontrol_partition.txt", r.stdout)

    # 6. slurmd / jobcomp logs in the job window --------------------------------
    start = parse_slurm_timestamp(parent.get("Start"))
    end = parse_slurm_timestamp(parent.get("End"))
    slurmd_text = _filter_log(SLURMD_LOG, jobid, start, end, cfg)
    save("slurmd.filtered.log", slurmd_text)
    jobcomp_text = _filter_log(JOBCOMP_LOG, jobid, None, None, cfg)
    save("jobcomp.filtered.log", jobcomp_text)

    # 7. dmesg (OOM killer, segfaults, GPU Xids) --------------------------------
    suspicious = parent.get("State", "").startswith("OUT_OF_MEMORY")
    dmesg_text = _collect_dmesg(nodes, cfg, record_cmd, force=suspicious)
    save("dmesg.txt", dmesg_text)

    # 8. GPU state ---------------------------------------------------------------
    tres = (parent.get("AllocTRES") or "") + (parent.get("ReqTRES") or "")
    if "gres/gpu" in tres:
        gpu_chunks = []
        for node in nodes:
            r = record_cmd(
                run(
                    _node_argv(
                        cfg,
                        node,
                        [
                            "nvidia-smi",
                            "--query-gpu=index,name,memory.used,memory.total",
                            "--format=csv",
                        ],
                    ),
                    timeout=cfg.cmd_timeout,
                )
            )
            gpu_chunks.append(f"### node {node} (rc={r.returncode})\n{r.stdout or r.stderr}")
        save("gpu.txt", "\n".join(gpu_chunks))

    # 9. stored job environment (AccountingStoreFlags=job_env), redacted --------
    r = record_cmd(run(["sacct", "-j", jobid, "--env-vars"], timeout=cfg.cmd_timeout))
    if r.ok and r.stdout.strip() and "invalid option" not in r.stderr.lower():
        save("env.redacted.txt", redact(r.stdout))

    manifest["parent_summary"] = {
        k: parent.get(k, "")
        for k in (
            "JobID", "JobName", "User", "Partition", "State", "ExitCode",
            "DerivedExitCode", "Reason", "Submit", "Start", "End", "Elapsed",
            "Timelimit", "ReqMem", "ReqCPUS", "AllocCPUS", "ReqTRES",
            "AllocTRES", "NodeList", "NNodes", "WorkDir", "Comment",
        )
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("collected job %s into %s (%d artifacts)", jobid, cache,
             len(manifest["artifacts"]))
    return Bundle(jobid, cache, manifest)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _collect_script(jobid, sc, cfg, record_cmd) -> str | None:
    """Submit script via sacct -B, falling back to scontrol -dd / Command=."""
    r = record_cmd(run(["sacct", "-j", jobid, "-B"], timeout=cfg.cmd_timeout))
    if r.ok and r.stdout.strip():
        text = r.stdout
        # strip the "Batch Script for <jobid>" banner sacct prints
        lines = text.splitlines()
        body_start = 0
        for i, ln in enumerate(lines[:4]):
            if set(ln.strip()) == {"-"} or ln.startswith("Batch Script"):
                body_start = i + 1
        body = "\n".join(lines[body_start:]).strip("\n")
        if body and body.upper() != "NONE":
            return body + "\n"
    # fallback 1: scontrol -dd output contains a BatchScript= section
    raw = sc.get("BatchScript")
    if raw:
        return raw if raw.endswith("\n") else raw + "\n"
    # fallback 2: read the Command= path from disk
    cmd = sc.get("Command")
    if cmd and cmd not in ("(null)", "None"):
        path = cmd.split()[0]
        try:
            return Path(path).read_text(errors="replace")
        except OSError:
            pass
    return None


def _io_paths(sc, script, parent, meta) -> tuple[str | None, str | None]:
    """Resolve stdout/stderr paths: scontrol first, script directives second."""
    stdout = sc.get("StdOut")
    stderr = sc.get("StdErr")
    if stdout and stderr:
        return stdout, stderr
    out_pat = err_pat = None
    for line in (script or "").splitlines():
        m = re.match(r"^#SBATCH\s+(?:-o|--output)[=\s]+(\S+)", line)
        if m:
            out_pat = m.group(1)
        m = re.match(r"^#SBATCH\s+(?:-e|--error)[=\s]+(\S+)", line)
        if m:
            err_pat = m.group(1)
    workdir = parent.get("WorkDir") or "."
    raw_id = parent.get("JobIDRaw") or parent.get("JobID") or meta["jobid"]
    meta = dict(meta, jobid=raw_id.split(".")[0])
    if not out_pat:
        out_pat = f"slurm-{meta['jobid']}.out"
    out = expand_filename_pattern(out_pat, meta)
    err = expand_filename_pattern(err_pat, meta) if err_pat else out
    def absolutize(p):
        return p if os.path.isabs(p) else os.path.join(workdir, p)
    return stdout or absolutize(out), stderr or absolutize(err)


def _expand_nodelist(nodelist: str, cfg, record_cmd) -> list[str]:
    nodelist = nodelist.strip()
    if not nodelist or nodelist in ("None assigned", "(null)"):
        return []
    if not re.search(r"[\[\],]", nodelist):
        return [nodelist]
    r = record_cmd(run(["scontrol", "show", "hostnames", nodelist], timeout=10))
    if r.ok:
        return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    return [nodelist]


_TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")


def _filter_log(path: str, jobid: str, start, end, cfg) -> str | None:
    """Keep log lines that reference the job id in a job context (StepId=N,
    [N.batch], JobId=N, "JOB N ON ..."); if a time window is known, also keep
    error-level lines inside it (±2 min)."""
    text, meta = read_file_capped(path, 64 * 1024 * 1024)
    if text is None:
        return None
    base = re.escape(jobid.split("_")[0].split(".")[0])
    # stepd prefix "[123.batch]", "JobId=123"/"job_id=123"/"StepId=123.x",
    # or prose "job 123" / "JOB 123 ON ..."
    id_re = re.compile(
        rf"(?i)(\[{base}\.[a-z0-9]+\]"
        rf"|(?:job_?id|stepid)=0*{base}(?![0-9])"
        rf"|\bjob\s+0*{base}(?![0-9]))"
    )
    win_lo = start - timedelta(seconds=120) if start else None
    win_hi = end + timedelta(seconds=120) if end else None
    keep: list[str] = []
    for ln in text.splitlines():
        if id_re.search(ln):
            keep.append(ln)
            continue
        if win_lo and ("error" in ln.lower() or "fatal" in ln.lower()):
            m = _TS_RE.match(ln)
            ts = parse_slurm_timestamp(m.group(1)) if m else None
            if ts and win_lo <= ts <= win_hi:
                keep.append(ln)
    if not keep:
        return None
    capped, _ = head_tail_cap("\n".join(keep) + "\n", cfg.max_io_bytes)
    return capped


def _collect_dmesg(nodes, cfg, record_cmd, force=False) -> str | None:
    """Grab OOM/segfault/GPU lines from dmesg on each node (best effort)."""
    pat = re.compile(
        r"(?i)out of memory|oom[-_ ]kill|killed process|segfault|"
        r"general protection|NVRM|Xid"
    )
    chunks = []
    targets = nodes or [None]
    seen_local = False
    for node in targets:
        argv = ["dmesg", "-T"]
        if node:
            argv = _node_argv(cfg, node, argv)
            if argv == ["dmesg", "-T"]:  # local mode: identical kernel, run once
                if seen_local:
                    continue
                seen_local = True
        r = record_cmd(run(argv, timeout=cfg.cmd_timeout))
        if not r.ok:
            continue
        hits = [ln for ln in r.stdout.splitlines() if pat.search(ln)]
        if hits:
            chunks.append(f"### dmesg {node or '(local)'}\n" + "\n".join(hits[-200:]))
    if not chunks and not force:
        return None
    return "\n".join(chunks) or "(no OOM/segfault/GPU lines found in dmesg)"
