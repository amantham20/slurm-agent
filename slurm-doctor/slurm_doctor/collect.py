"""Gather everything we need to diagnose a failed SLURM job — once, idempotently.

Phase 1 of the slurm-doctor pipeline. Pulls accounting, the original submit
script, stdout/stderr, per-node state and slurmd log slices, dmesg (when memory
looks tight or the state says OOM), and GPU state (when AllocTRES has gres/gpu).

Every artifact is cached under ``~/.cache/slurm-doctor/<jobid>/`` as raw text so
re-running the parser doesn't re-hit sacct.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from . import SCHEMA_VERSION
from ._shell import ShellResult, default_runner

log = logging.getLogger("slurm_doctor.collect")

SACCT_FORMAT = (
    "JobID,JobIDRaw,JobName,User,Partition,State,ExitCode,DerivedExitCode,Reason,"
    "Submit,Start,End,Elapsed,Timelimit,ReqMem,MaxRSS,MaxVMSize,AveCPU,"
    "ReqCPUS,AllocCPUS,ReqTRES,AllocTRES,NodeList,NNodes,WorkDir"
)
SACCT_COLS = SACCT_FORMAT.split(",")

DEFAULT_MAX_LOG_BYTES = 4 * 1024 * 1024  # 4 MiB
SLURMD_LOG_WINDOW_SLOP_S = 60  # widen the time window when slicing slurmd.log


# ---------------------------------------------------------------------------
# Bundle shape (the on-disk index + a friendly Python view)
# ---------------------------------------------------------------------------

@dataclass
class StdioFile:
    declared_path: str | None
    cached_path: str | None
    original_size: int | None = None
    truncated: bool = False


@dataclass
class CollectedBundle:
    jobid: str
    cache_dir: str
    schema_version: int = SCHEMA_VERSION

    # Headline accounting fields (parent step), populated when available
    state: str | None = None
    jobname: str | None = None
    exit_code: str | None = None
    derived_exit_code: str | None = None
    reason: str | None = None
    nodelist: str | None = None
    workdir: str | None = None
    req_mem: str | None = None
    max_rss: str | None = None  # max across all steps
    elapsed: str | None = None
    timelimit: str | None = None
    alloc_tres: str | None = None
    submit_time: str | None = None
    start_time: str | None = None
    end_time: str | None = None

    accounting_steps: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)              # parsed scontrol
    submit_script_path: str | None = None
    stdout: StdioFile | None = None
    stderr: StdioFile | None = None
    nodes: dict[str, dict] = field(default_factory=dict)  # nodename -> {state_path, slurmd_log_path}
    dmesg_path: str | None = None
    gpu: dict[str, dict] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # asdict turns nested dataclasses to dicts already; nothing else to do
        return d


# ---------------------------------------------------------------------------
# Small parsers — kept pure so they're trivial to unit test
# ---------------------------------------------------------------------------

def parse_sacct_parsable2(text: str, columns: list[str]) -> list[dict[str, str]]:
    """Parse `sacct --parsable2` output: pipe-separated, one row per step.

    The first line may or may not be a header (when called with `-n` it's not).
    We detect by matching the first cell against a known column name.
    """
    rows: list[dict[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue
        # --parsable2 leaves no trailing | (vs --parsable which does)
        parts = line.split("|")
        # Skip a header row if present.
        if rows == [] and parts and parts[0] == columns[0]:
            continue
        # Pad/truncate to column count.
        if len(parts) < len(columns):
            parts = parts + [""] * (len(columns) - len(parts))
        elif len(parts) > len(columns):
            parts = parts[: len(columns)]
        rows.append(dict(zip(columns, parts)))
    return rows


_SCONTROL_KV = re.compile(r"(\S+?)=(.+?)(?=\s+\S+?=|$)")


def parse_scontrol_kv(text: str) -> dict[str, str]:
    """Parse `scontrol show job/node` key=value output across multiple lines.

    Whitespace separates KEY=VALUE pairs; values may contain spaces only if
    they're bracketed (lists). For our purposes we use a regex that consumes
    everything up to the next `KEY=`.
    """
    flat = " ".join(line.strip() for line in text.splitlines() if line.strip())
    out: dict[str, str] = {}
    for m in _SCONTROL_KV.finditer(flat):
        key = m.group(1)
        value = m.group(2).strip()
        # If the same key shows up twice (multi-step), keep the first.
        out.setdefault(key, value)
    return out


def _ratio(maxrss: str | None, reqmem: str | None) -> float | None:
    """Return MaxRSS / ReqMem as a float, or None if either can't be parsed."""
    def _bytes(v: str | None) -> int | None:
        if not v:
            return None
        v = v.strip()
        m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*([KMGTPE]?)$", v, re.IGNORECASE)
        if not m:
            # ReqMem can be "4000M", "16Gn", "16Gc" — strip a trailing 'n'/'c'
            m2 = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*([KMGTPE])[nc]?$", v, re.IGNORECASE)
            if not m2:
                return None
            m = m2
        n = float(m.group(1))
        unit = (m.group(2) or "").upper()
        mult = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5, "E": 1024**6}[unit]
        return int(n * mult)

    mr = _bytes(maxrss)
    rq = _bytes(reqmem)
    if mr is None or rq is None or rq == 0:
        return None
    return mr / rq


def parse_slurm_time(s: str | None) -> datetime | None:
    if not s or s in ("Unknown", "None"):
        return None
    # SLURM uses "YYYY-MM-DDTHH:MM:SS"
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class Collector:
    def __init__(
        self,
        jobid: str,
        cache_root: str | os.PathLike[str] | None = None,
        *,
        runner: Callable[..., ShellResult] = default_runner,
        max_log_bytes: int = DEFAULT_MAX_LOG_BYTES,
        slurmd_log_candidates: list[str] | None = None,
        refresh: bool = True,
    ) -> None:
        self.jobid = str(jobid)
        if not re.match(r"^[0-9]+(_[0-9]+)?$", self.jobid):
            raise ValueError(f"unsafe jobid: {self.jobid!r}")
        root = Path(cache_root) if cache_root else Path.home() / ".cache" / "slurm-doctor"
        self.cache_dir = root / self.jobid
        self.run = runner
        self.max_log_bytes = max_log_bytes
        self.refresh = refresh
        self.slurmd_log_candidates = slurmd_log_candidates or [
            "/var/log/slurm/slurmd-{node}.log",
            "/var/log/slurm/slurmd.log",
        ]

    # -- top-level orchestrator --------------------------------------------

    def collect(self) -> CollectedBundle:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        bundle = CollectedBundle(jobid=self.jobid, cache_dir=str(self.cache_dir))

        # 1. Accounting (sacct).
        try:
            self._accounting(bundle)
        except Exception as e:  # noqa: BLE001 - surface as recoverable
            bundle.errors.append(f"accounting: {e!r}")

        # 2. scontrol show job for paths/work dirs/extra metadata.
        try:
            self._meta(bundle)
        except Exception as e:  # noqa: BLE001
            bundle.errors.append(f"meta: {e!r}")

        # 3. Submit script.
        try:
            self._submit_script(bundle)
        except Exception as e:  # noqa: BLE001
            bundle.errors.append(f"submit_script: {e!r}")

        # 4. Stdout/Stderr (head+tail truncated).
        try:
            self._stdio(bundle)
        except Exception as e:  # noqa: BLE001
            bundle.errors.append(f"stdio: {e!r}")

        # 5. Per-node state and slurmd log slice.
        try:
            self._nodes(bundle)
        except Exception as e:  # noqa: BLE001
            bundle.errors.append(f"nodes: {e!r}")

        # 6. dmesg if memory looks suspicious.
        try:
            self._maybe_dmesg(bundle)
        except Exception as e:  # noqa: BLE001
            bundle.errors.append(f"dmesg: {e!r}")

        # 7. GPU state if AllocTRES mentions gres/gpu.
        try:
            self._maybe_gpu(bundle)
        except Exception as e:  # noqa: BLE001
            bundle.errors.append(f"gpu: {e!r}")

        # Persist the bundle index for the parser to consume.
        self._write_bundle_index(bundle)
        return bundle

    # -- steps --------------------------------------------------------------

    def _accounting(self, bundle: CollectedBundle) -> None:
        argv = ["sacct", "-j", self.jobid, "--parsable2", f"--format={SACCT_FORMAT}"]
        r = self.run(argv, timeout=30)
        raw = r.stdout
        (self.cache_dir / "accounting.txt").write_text(raw)
        if r.returncode != 0 or r.missing:
            bundle.errors.append(f"sacct rc={r.returncode}, stderr={r.stderr[:200]}")
            return
        steps = parse_sacct_parsable2(raw, SACCT_COLS)
        bundle.accounting_steps = steps
        if not steps:
            return
        parent = steps[0]  # parent row is JobID == self.jobid (no .step suffix)
        bundle.state = parent.get("State") or None
        bundle.jobname = parent.get("JobName") or None
        bundle.exit_code = parent.get("ExitCode") or None
        bundle.derived_exit_code = parent.get("DerivedExitCode") or None
        bundle.reason = parent.get("Reason") or None
        bundle.nodelist = parent.get("NodeList") or None
        bundle.workdir = parent.get("WorkDir") or None
        bundle.req_mem = parent.get("ReqMem") or None
        bundle.elapsed = parent.get("Elapsed") or None
        bundle.timelimit = parent.get("Timelimit") or None
        bundle.alloc_tres = parent.get("AllocTRES") or None
        bundle.submit_time = parent.get("Submit") or None
        bundle.start_time = parent.get("Start") or None
        bundle.end_time = parent.get("End") or None
        # MaxRSS lives on step rows (.batch, .0, ...). Take the max non-empty.
        rss_vals = [s.get("MaxRSS", "") for s in steps if s.get("MaxRSS")]
        bundle.max_rss = max(rss_vals, key=lambda v: _bytes_safe(v)) if rss_vals else None

    def _meta(self, bundle: CollectedBundle) -> None:
        argv = ["scontrol", "show", "job", self.jobid, "-dd"]
        r = self.run(argv, timeout=15)
        (self.cache_dir / "scontrol.txt").write_text(r.stdout)
        if r.returncode != 0 or r.missing:
            # When the job is purged from controller memory, scontrol fails.
            # That's fine — accounting (sacct/dbd) still gave us most fields.
            bundle.errors.append(f"scontrol rc={r.returncode}")
            return
        bundle.meta = parse_scontrol_kv(r.stdout)
        # Backfill WorkDir/NodeList from scontrol if accounting was empty.
        if not bundle.workdir:
            bundle.workdir = bundle.meta.get("WorkDir") or bundle.workdir
        if not bundle.nodelist:
            bundle.nodelist = bundle.meta.get("NodeList") or bundle.nodelist

    def _submit_script(self, bundle: CollectedBundle) -> None:
        # Primary: sacct -B prints the submit script with a 2-line header.
        argv = ["sacct", "-B", "-j", self.jobid]
        r = self.run(argv, timeout=15)
        script = r.stdout
        # Strip the "Batch Script for NNN\n----...\n" header if present.
        lines = script.splitlines(keepends=True)
        if len(lines) >= 2 and lines[1].lstrip().startswith("---"):
            script = "".join(lines[2:])
        # sacct returns the literal "NONE" when AccountingStoreFlags=job_script
        # is not enabled — treat it as missing and trigger the fallback.
        if script.strip() in ("", "NONE"):
            script = ""
        # Fallback 1: read Command= path from scontrol meta if -B produced nothing.
        if not script.strip():
            cmd_path = bundle.meta.get("Command")
            if cmd_path and Path(cmd_path).is_file():
                script = Path(cmd_path).read_text()
        # Fallback 2 (job purged from controller AND no -B): guess the script
        # from WorkDir + JobName, e.g. /data/oom.sh for JobName=oom.
        if not script.strip():
            script = self._guess_script_from_workdir(bundle)
        if script.strip():
            dst = self.cache_dir / "submit_script.sh"
            dst.write_text(script)
            bundle.submit_script_path = str(dst)

    def _guess_script_from_workdir(self, bundle: CollectedBundle) -> str:
        """Last-resort submit-script recovery: <WorkDir>/<JobName>[.sh]."""
        wd = bundle.workdir
        name = bundle.jobname
        if not wd or not name or not Path(wd).is_dir():
            return ""
        for cand in (Path(wd) / f"{name}.sh", Path(wd) / name, Path(wd) / f"{name}.sbatch"):
            if cand.is_file():
                try:
                    return cand.read_text(errors="replace")
                except OSError:
                    continue
        return ""

    def _stdio(self, bundle: CollectedBundle) -> None:
        # Primary source: scontrol's StdOut/StdErr. Fallbacks for a purged job:
        #  (a) #SBATCH --output/--error directives from the submit script,
        #  (b) glob WorkDir for files whose name contains the jobid.
        from_script = _sbatch_io_paths(bundle.submit_script_path, bundle.jobid)
        glob_io = self._glob_workdir_io(bundle)
        for stream, key in (("StdOut", "output"), ("StdErr", "error")):
            declared = bundle.meta.get(stream) or from_script.get(key) or glob_io.get(key)
            if not declared:
                continue
            kind = stream.lower()
            target = self.cache_dir / f"{kind}.txt"
            src = Path(declared)
            f = StdioFile(declared_path=declared, cached_path=None)
            if src.is_file():
                size, truncated = truncated_copy(src, target, self.max_log_bytes)
                f.original_size = size
                f.truncated = truncated
                f.cached_path = str(target)
            setattr(bundle, kind, f)

    def _glob_workdir_io(self, bundle: CollectedBundle) -> dict[str, str]:
        """Best-effort: find <WorkDir>/*<jobid>*.out / .err for a purged job."""
        wd = bundle.workdir
        if not wd or not Path(wd).is_dir():
            return {}
        out: dict[str, str] = {}
        try:
            entries = list(Path(wd).iterdir())
        except OSError:
            return {}
        for key, exts in (("output", (".out", ".log")), ("error", (".err",))):
            for p in entries:
                if self.jobid in p.name and p.suffix in exts and p.is_file():
                    out[key] = str(p)
                    break
        return out

    def _nodes(self, bundle: CollectedBundle) -> None:
        if not bundle.nodelist or bundle.nodelist in ("None assigned", ""):
            return
        nodes = expand_nodelist(bundle.nodelist)
        (self.cache_dir / "nodes").mkdir(exist_ok=True)
        for n in nodes:
            entry: dict[str, Any] = {}
            r = self.run(["scontrol", "show", "node", n], timeout=10)
            state_path = self.cache_dir / "nodes" / f"{n}.state.txt"
            state_path.write_text(r.stdout)
            entry["state_path"] = str(state_path)
            if r.returncode == 0:
                entry["state"] = parse_scontrol_kv(r.stdout)
            # slurmd log slice — best-effort across candidate paths.
            slurmd_lines = self._slurmd_log_slice(n, bundle.start_time, bundle.end_time)
            if slurmd_lines:
                log_path = self.cache_dir / "nodes" / f"{n}.slurmd.log"
                log_path.write_text(slurmd_lines)
                entry["slurmd_log_path"] = str(log_path)
            bundle.nodes[n] = entry

    def _slurmd_log_slice(self, node: str, start: str | None, end: str | None) -> str:
        start_dt = parse_slurm_time(start)
        end_dt = parse_slurm_time(end)
        for tmpl in self.slurmd_log_candidates:
            path = Path(tmpl.format(node=node))
            if not path.is_file():
                continue
            try:
                content = path.read_text(errors="replace")
            except OSError:
                continue
            return _slice_log_by_time(content, start_dt, end_dt, jobid=self.jobid)
        return ""

    def _maybe_dmesg(self, bundle: CollectedBundle) -> None:
        # Trigger: state says OOM, or MaxRSS is suspiciously close to ReqMem.
        trigger = (bundle.state or "").upper() in ("OUT_OF_MEMORY", "OOM_KILLED")
        ratio = _ratio(bundle.max_rss, bundle.req_mem)
        if not trigger and (ratio is None or ratio < 0.9):
            return
        r = self.run(["dmesg", "-T"], timeout=10)
        if r.missing or r.returncode != 0:
            return
        keep = [
            ln for ln in r.stdout.splitlines()
            if re.search(r"(?i)killed process|out of memory|oom-killer|invoked oom", ln)
        ]
        if not keep:
            return
        path = self.cache_dir / "dmesg.txt"
        path.write_text("\n".join(keep) + "\n")
        bundle.dmesg_path = str(path)

    def _maybe_gpu(self, bundle: CollectedBundle) -> None:
        if not bundle.alloc_tres or "gres/gpu" not in bundle.alloc_tres:
            return
        r = self.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv"],
            timeout=10,
        )
        if r.missing:
            bundle.errors.append("gpu: nvidia-smi missing on slurmctld; collect from node")
            return
        if r.returncode != 0:
            return
        (self.cache_dir / "nvidia-smi.csv").write_text(r.stdout)
        bundle.gpu["slurmctld"] = {"path": str(self.cache_dir / "nvidia-smi.csv")}

    # -- index --------------------------------------------------------------

    def _write_bundle_index(self, bundle: CollectedBundle) -> None:
        (self.cache_dir / "bundle.json").write_text(
            json.dumps(bundle.to_dict(), indent=2, default=str)
        )


# ---------------------------------------------------------------------------
# Helpers — also exported for testing
# ---------------------------------------------------------------------------

def _bytes_safe(v: str) -> int:
    """Lenient parse of MaxRSS-style suffixes for sorting only."""
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*([KMGTPE]?)", v.strip(), re.IGNORECASE)
    if not m:
        return 0
    n = float(m.group(1))
    unit = (m.group(2) or "").upper()
    return int(n * {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3,
                    "T": 1024**4, "P": 1024**5, "E": 1024**6}[unit])


def truncated_copy(src: Path, dst: Path, max_bytes: int) -> tuple[int, bool]:
    """Copy `src` to `dst`, keeping ``max_bytes/2`` from each end if oversize."""
    size = src.stat().st_size
    if size <= max_bytes:
        dst.write_bytes(src.read_bytes())
        return size, False
    head_n = max_bytes // 2
    tail_n = max_bytes - head_n
    with src.open("rb") as f:
        head = f.read(head_n)
        f.seek(size - tail_n)
        tail = f.read(tail_n)
    marker = f"\n[... {size - max_bytes} bytes truncated by slurm-doctor ...]\n".encode()
    dst.write_bytes(head + marker + tail)
    return size, True


_NODELIST_RANGE = re.compile(r"([a-zA-Z][a-zA-Z0-9_-]*)\[([0-9,\-]+)\]")

_SBATCH_OUTPUT = re.compile(
    r"^\s*#SBATCH\s+(?:--(output|error)[=\s]+|-o\s+|-e\s+)([^\s#]+)",
    re.MULTILINE,
)


def _sbatch_io_paths(script_path: str | None, jobid: str) -> dict[str, str]:
    """Extract output / error paths from #SBATCH directives, substituting %j.

    Only handles the common substitutions (%j, %J, %x is intentionally skipped
    because we don't have JobName here). Returns {} when the script isn't
    available.
    """
    if not script_path:
        return {}
    try:
        text = Path(script_path).read_text(errors="replace")
    except OSError:
        return {}
    out: dict[str, str] = {}
    for m in _SBATCH_OUTPUT.finditer(text):
        flag, path = m.group(1), m.group(2)
        key = (
            flag
            if flag in ("output", "error")
            else ("output" if "-o" in m.group(0) else "error")
        )
        path = path.replace("%j", jobid).replace("%J", jobid)
        out.setdefault(key, path)
    return out


def expand_nodelist(nodelist: str) -> list[str]:
    """Expand SLURM nodelist strings like 'c[1-3,5]' into ['c1','c2','c3','c5'].

    Bare nodes and comma-separated lists outside brackets are also handled.
    """
    if not nodelist:
        return []
    out: list[str] = []
    # Split on commas that are NOT inside brackets.
    parts: list[str] = []
    depth = 0
    cur = ""
    for ch in nodelist:
        if ch == "[":
            depth += 1
            cur += ch
        elif ch == "]":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur:
        parts.append(cur)
    for part in parts:
        part = part.strip()
        m = _NODELIST_RANGE.match(part)
        if not m:
            if part:
                out.append(part)
            continue
        prefix, ranges = m.group(1), m.group(2)
        for r in ranges.split(","):
            if "-" in r:
                a, b = r.split("-", 1)
                width = max(len(a), len(b))
                for i in range(int(a), int(b) + 1):
                    out.append(f"{prefix}{str(i).zfill(width if a.startswith('0') else 1)}")
            else:
                out.append(f"{prefix}{r}")
    return out


_SLURMD_TS = re.compile(r"^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.\d+)?\]")


def _slice_log_by_time(content: str, start: datetime | None, end: datetime | None, *, jobid: str) -> str:
    """Return only the lines whose [timestamp] falls inside [start-slop, end+slop],
    OR which mention the job id explicitly.

    If neither bound is known, returns lines mentioning the jobid.
    """
    from datetime import timedelta
    slop = timedelta(seconds=SLURMD_LOG_WINDOW_SLOP_S)
    keep: list[str] = []
    jobtok = f"JobId={jobid}"
    step_prefix = f"[{jobid}."  # slurmd step lines: [2.batch], [2.0], ...
    for line in content.splitlines():
        if jobtok in line or step_prefix in line:
            keep.append(line)
            continue
        if start is None or end is None:
            continue
        m = _SLURMD_TS.match(line)
        if not m:
            continue
        try:
            t = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
        if start - slop <= t <= end + slop:
            keep.append(line)
    return "\n".join(keep)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def collect(jobid: str | int, **kwargs) -> CollectedBundle:
    return Collector(str(jobid), **kwargs).collect()
