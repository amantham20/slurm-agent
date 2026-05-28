"""Single shell-out wrapper used by every collector/diagnoser path.

Goals:
- never use shell=True with interpolation (everything is argv-list)
- always enforce a timeout
- always capture stdout and stderr
- log the invocation for traceability
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("slurm_doctor.shell")

DEFAULT_TIMEOUT_S = 30


@dataclass
class ShellResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    missing: bool = False  # binary not on PATH


def _exec(
    argv: list[str],
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
) -> ShellResult:
    if not argv:
        raise ValueError("argv must be non-empty")
    if shutil.which(argv[0]) is None and not Path(argv[0]).is_file():
        log.debug("binary missing: %s", argv[0])
        return ShellResult(argv=list(argv), returncode=127, stdout="", stderr="", missing=True)
    log.debug("$ %s", " ".join(argv))
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            input=input_text,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        log.warning("timeout after %ss: %s", timeout, " ".join(argv))
        return ShellResult(
            argv=list(argv),
            returncode=-1,
            stdout=(e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")),
            stderr=(e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")),
            timed_out=True,
        )
    return ShellResult(
        argv=list(argv),
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )


Runner = "callable[[list[str]], ShellResult]"


def default_runner(argv: list[str], *, timeout: float = DEFAULT_TIMEOUT_S, **kw) -> ShellResult:
    """Module-level default runner; tests inject a fake."""
    return _exec(argv, timeout=timeout, **kw)
