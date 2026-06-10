"""Runtime configuration, sourced from environment variables and CLI flags.

slurm-doctor keeps no state beyond two directories:

* the cache dir (raw collected artifacts, one subdir per job id)
* the report dir (rendered reports + the heal chain ledger)
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

DEFAULT_REPORT_DIR = "/data/jobs/.slurm-doctor"
DEFAULT_AUTOFIX_KINDS = ("bump_memory", "bump_time", "add_set_eux")
# Fix kinds that may make things worse than the original failure; these
# always require an explicit --yes for `heal`, regardless of confidence.
RISKY_FIX_KINDS = frozenset(
    {"swap_mpi_launcher", "pin_gpu_visible", "request_constraint"}
)
FAILURE_STATES = (
    "FAILED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "BOOT_FAIL",
    "DEADLINE",
    "CANCELLED",
    "PREEMPTED",
)


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclasses.dataclass
class Config:
    cache_dir: Path
    report_dir: Path
    max_io_bytes: int = 4 * 1024 * 1024  # per stdout/stderr file
    cmd_timeout: float = 30.0
    llm_enabled: bool = False
    llm_model: str = "claude-sonnet-4-5"
    autofix_kinds: tuple[str, ...] = DEFAULT_AUTOFIX_KINDS
    heal_confidence: float = 0.85
    max_heal_chain: int = 2  # refuse to heal a job already healed this many times
    restd_url: str | None = None  # e.g. http://slurmrestd:6820 or unix:///path.sock
    node_exec: str | None = None  # e.g. 'ssh {node}' to run node commands remotely

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        env = os.environ
        cfg = cls(
            cache_dir=Path(
                env.get("SLURM_DOCTOR_CACHE")
                or os.path.join(
                    env.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
                    "slurm-doctor",
                )
            ),
            report_dir=Path(env.get("SLURM_DOCTOR_REPORT_DIR", DEFAULT_REPORT_DIR)),
            max_io_bytes=int(env.get("SLURM_DOCTOR_MAX_IO_BYTES", 4 * 1024 * 1024)),
            cmd_timeout=float(env.get("SLURM_DOCTOR_CMD_TIMEOUT", 30)),
            llm_enabled=_env_flag("SLURM_DOCTOR_LLM"),
            llm_model=env.get("SLURM_DOCTOR_LLM_MODEL", "claude-sonnet-4-5"),
            autofix_kinds=tuple(
                k.strip()
                for k in env.get(
                    "SLURM_DOCTOR_AUTOFIX_KINDS", ",".join(DEFAULT_AUTOFIX_KINDS)
                ).split(",")
                if k.strip()
            ),
            heal_confidence=float(env.get("SLURM_DOCTOR_HEAL_CONFIDENCE", 0.85)),
            max_heal_chain=int(env.get("SLURM_DOCTOR_MAX_HEAL_CHAIN", 2)),
            restd_url=env.get("SLURM_DOCTOR_RESTD") or None,
            node_exec=env.get("SLURM_DOCTOR_NODE_EXEC") or None,
        )
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg

    def job_cache(self, jobid: str) -> Path:
        return self.cache_dir / str(jobid)

    def job_report_dir(self, jobid: str) -> Path:
        return self.report_dir / str(jobid)
