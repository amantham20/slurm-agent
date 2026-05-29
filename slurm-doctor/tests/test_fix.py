"""Tests for fix.py — patchers + filename + idempotency + over-edit guard."""
from __future__ import annotations

from pathlib import Path

import pytest

from slurm_doctor.collect import CollectedBundle
from slurm_doctor.fix import (
    FixError,
    apply_fix,
    _format_duration,
    _parse_duration,
    _replace_or_insert_sbatch,
    write_patched_script,
)


SCRIPT = (
    "#!/bin/bash\n"
    "#SBATCH --job-name=oom\n"
    "#SBATCH --output=/data/oom_%j.out\n"
    "#SBATCH --error=/data/oom_%j.err\n"
    "#SBATCH --mem=200M\n"
    "#SBATCH --time=00:01:00\n"
    "ulimit -v $(( ${SLURM_MEM_PER_NODE:-100} * 1024 ))\n"
    "python3 -c 'pass'\n"
)


def _bundle(**kw) -> CollectedBundle:
    b = CollectedBundle(jobid="1", cache_dir="/tmp")
    for k, v in kw.items():
        setattr(b, k, v)
    return b


# ---- bump_memory ----------------------------------------------------------

def test_bump_memory_replaces_mem_with_ceil_30pct_to_GB():
    # 200M used * 1.3 = 260M -> ceil to next GB = 1G
    b = _bundle(max_rss="200M", workdir="/data")
    out = apply_fix("bump_memory", SCRIPT, b)
    assert out.patched_text is not None
    assert "--mem=1G" in out.patched_text
    # original value gone
    assert "--mem=200M" not in out.patched_text
    # diff present
    assert "--- original" in out.diff
    assert "+++ patched (bump_memory)" in out.diff


def test_bump_memory_handles_GB_input():
    # 1.5G used * 1.3 = 1.95G -> ceil to 2G
    b = _bundle(max_rss="1500M", workdir="/data")
    out = apply_fix("bump_memory", SCRIPT, b)
    assert "--mem=2G" in out.patched_text


def test_bump_memory_skips_when_no_maxrss():
    b = _bundle(max_rss=None, workdir="/data")
    out = apply_fix("bump_memory", SCRIPT, b)
    assert out.patched_text is None
    assert "MaxRSS" in (out.skipped_reason or "")


# ---- bump_time ------------------------------------------------------------

def test_bump_time_rounds_up_to_minute():
    # elapsed=00:00:30 * 1.5 = 45s -> rounded up to 1 minute -> 00:01:00
    b = _bundle(elapsed="00:00:30", workdir="/data")
    out = apply_fix("bump_time", SCRIPT, b)
    assert out.patched_text is not None
    assert "--time=00:01:00" in out.patched_text


def test_bump_time_long_elapsed():
    # elapsed=00:30:00 * 1.5 = 45min -> 00:45:00
    b = _bundle(elapsed="00:30:00", workdir="/data")
    out = apply_fix("bump_time", SCRIPT, b)
    assert "--time=00:45:00" in out.patched_text


def test_bump_time_dashed_days():
    # elapsed=1-00:00:00 (1 day) * 1.5 = 1.5 days = 1-12:00:00
    b = _bundle(elapsed="1-00:00:00", workdir="/data")
    out = apply_fix("bump_time", SCRIPT, b)
    assert "--time=1-12:00:00" in out.patched_text


# ---- add_set_eux ----------------------------------------------------------

def test_add_set_eux_inserts_after_sbatch_block():
    b = _bundle(workdir="/data")
    out = apply_fix("add_set_eux", SCRIPT, b)
    assert out.patched_text is not None
    assert "set -euo pipefail" in out.patched_text
    # Inserted BEFORE the first command (ulimit line), AFTER SBATCH block.
    idx_set = out.patched_text.index("set -euo pipefail")
    idx_ulimit = out.patched_text.index("ulimit -v")
    idx_last_sbatch = out.patched_text.rindex("#SBATCH")
    assert idx_last_sbatch < idx_set < idx_ulimit


def test_add_set_eux_is_idempotent():
    script = "#!/bin/bash\nset -euo pipefail\necho hi\n"
    b = _bundle(workdir="/data")
    out = apply_fix("add_set_eux", script, b)
    assert out.patched_text is None
    assert "already" in (out.skipped_reason or "")


# ---- prepend_modules ------------------------------------------------------

def test_prepend_modules_sources_lmod_and_leaves_placeholder():
    b = _bundle(workdir="/data")
    out = apply_fix("prepend_modules", SCRIPT, b)
    assert out.patched_text is not None
    assert "source /etc/profile.d/lmod.sh" in out.patched_text
    assert "module load <NAME>" in out.patched_text


def test_prepend_modules_idempotent_when_already_loaded():
    script = (
        "#!/bin/bash\n#SBATCH -J x\n"
        "module load python/3.11\n"
        "python3 -c 'pass'\n"
    )
    out = apply_fix("prepend_modules", script, _bundle())
    assert out.patched_text is None
    assert "already" in (out.skipped_reason or "")


# ---- _replace_or_insert_sbatch -------------------------------------------

def test_replace_sbatch_with_equals():
    s = "#!/bin/bash\n#SBATCH --mem=200M\necho hi\n"
    out = _replace_or_insert_sbatch(s, "mem", "1G")
    assert "#SBATCH --mem=1G\n" in out


def test_replace_sbatch_with_space():
    s = "#!/bin/bash\n#SBATCH --mem 200M\necho hi\n"
    out = _replace_or_insert_sbatch(s, "mem", "1G")
    assert "#SBATCH --mem 1G\n" in out


def test_inserts_when_missing():
    s = "#!/bin/bash\n#SBATCH --time=00:01:00\necho hi\n"
    out = _replace_or_insert_sbatch(s, "mem", "1G")
    assert "#SBATCH --mem=1G" in out


# ---- duration utils -------------------------------------------------------

def test_duration_roundtrip():
    for s in ("00:00:30", "00:30:00", "01:30:00", "2-00:00:00", "0-01:00:00"):
        sec = _parse_duration(s)
        assert sec is not None and sec > 0


def test_format_duration_handles_days():
    assert _format_duration(86400) == "1-00:00:00"
    assert _format_duration(60 * 95) == "01:35:00"


# ---- write_patched_script -------------------------------------------------

def test_write_patched_script_increments_suffix(tmp_path):
    b = _bundle(workdir=str(tmp_path))
    b.submit_script_path = str(tmp_path / "oom.sh")
    Path(b.submit_script_path).write_text(SCRIPT)
    b.max_rss = "200M"
    out = apply_fix("bump_memory", SCRIPT, b)
    p1 = write_patched_script(b, out)
    p2 = write_patched_script(b, out)
    assert p1.name == "oom.fix1.sh"
    assert p2.name == "oom.fix2.sh"
    assert p1.stat().st_mode & 0o111  # executable


# ---- over-edit guard ------------------------------------------------------

def test_over_edit_downgrades_to_skipped():
    # Force a giant rewrite ratio by stuffing a fake patcher in apply_fix's
    # dispatch table via monkey-patching. We just exercise the public API.
    from slurm_doctor import fix as fix_mod

    def evil_patcher(text, bundle):
        return fix_mod.FixOutcome(
            fix_kind="evil", patched_text="line\n" * 100,
            skipped_reason=None, diff=None,
        )

    fix_mod._DISPATCH["__evil"] = evil_patcher
    try:
        out = apply_fix("__evil", "single original line\n", _bundle())
    finally:
        del fix_mod._DISPATCH["__evil"]
    assert out.patched_text is None
    assert "rewrite" in (out.skipped_reason or "")


# ---- unknown fix_kind -----------------------------------------------------

def test_unknown_fix_kind_returns_outcome_not_exception():
    out = apply_fix("totally_made_up", SCRIPT, _bundle())
    assert out.patched_text is None
    assert "not implemented" in (out.skipped_reason or "")


# ---- fix_path -------------------------------------------------------------

def test_fix_path_rewrites_to_close_match(tmp_path):
    from slurm_doctor.collect import StdioFile
    # WorkDir has "run_sim.py"; script calls the typo'd "./run_simm.py"
    (tmp_path / "run_sim.py").write_text("print('hi')\n")
    err = tmp_path / "stderr.txt"
    err.write_text("/bin/bash: line 8: ./run_simm.py: No such file or directory\n")
    script = "#!/bin/bash\n#SBATCH -J x\n./run_simm.py --flag\n"
    b = _bundle(workdir=str(tmp_path))
    b.stderr = StdioFile(declared_path="x", cached_path=str(err))
    out = apply_fix("fix_path", script, b)
    assert out.patched_text is not None
    assert "./run_sim.py --flag" in out.patched_text
    assert "run_simm.py" not in out.patched_text


def test_fix_path_skips_when_no_similar_file(tmp_path):
    from slurm_doctor.collect import StdioFile
    err = tmp_path / "stderr.txt"
    err.write_text("/bin/bash: line 8: ./totally_unrelated_xyz: No such file or directory\n")
    (tmp_path / "alpha.txt").write_text("")
    script = "#!/bin/bash\n./totally_unrelated_xyz\n"
    b = _bundle(workdir=str(tmp_path))
    b.stderr = StdioFile(declared_path="x", cached_path=str(err))
    out = apply_fix("fix_path", script, b)
    assert out.patched_text is None
    assert "similar file" in (out.skipped_reason or "")


# ---- swap_mpi_launcher ----------------------------------------------------

def test_swap_mpi_launcher_replaces_mpirun_and_np():
    script = "#!/bin/bash\n#SBATCH -J x\nmpirun -np 4 ./solver\n"
    out = apply_fix("swap_mpi_launcher", script, _bundle())
    assert out.patched_text is not None
    assert "srun --mpi=pmix -n 4 ./solver" in out.patched_text
    assert "mpirun" not in out.patched_text


def test_swap_mpi_launcher_idempotent():
    script = "#!/bin/bash\nsrun --mpi=pmix -n 4 ./solver\n"
    out = apply_fix("swap_mpi_launcher", script, _bundle())
    assert out.patched_text is None


def test_swap_mpi_leaves_comments_alone():
    script = "#!/bin/bash\n# mpirun is the old way\nmpiexec ./a.out\n"
    out = apply_fix("swap_mpi_launcher", script, _bundle())
    assert "# mpirun is the old way" in out.patched_text
    assert "srun --mpi=pmix ./a.out" in out.patched_text


# ---- request_constraint ---------------------------------------------------

def test_request_constraint_inserts_placeholder():
    out = apply_fix("request_constraint", SCRIPT, _bundle())
    assert out.patched_text is not None
    assert "#SBATCH --constraint=<FEATURE>" in out.patched_text


def test_request_constraint_idempotent():
    script = "#!/bin/bash\n#SBATCH --constraint=a100\necho hi\n"
    out = apply_fix("request_constraint", script, _bundle())
    assert out.patched_text is None


# ---- add_requeue_guard ----------------------------------------------------

def test_add_requeue_guard_adds_directive_and_backoff():
    out = apply_fix("add_requeue_guard", SCRIPT, _bundle())
    assert out.patched_text is not None
    assert "#SBATCH --requeue" in out.patched_text
    assert "SLURM_RESTART_COUNT" in out.patched_text


def test_add_requeue_guard_idempotent():
    script = (
        "#!/bin/bash\n#SBATCH --requeue\n"
        'if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then sleep 5; fi\n'
        "echo hi\n"
    )
    out = apply_fix("add_requeue_guard", script, _bundle())
    assert out.patched_text is None


# ---- pin_gpu_visible ------------------------------------------------------

def test_pin_gpu_visible_exports_cuda_visible_devices():
    out = apply_fix("pin_gpu_visible", SCRIPT, _bundle())
    assert out.patched_text is not None
    assert "CUDA_VISIBLE_DEVICES" in out.patched_text


def test_pin_gpu_visible_idempotent():
    script = "#!/bin/bash\nexport CUDA_VISIBLE_DEVICES=0\n./train\n"
    out = apply_fix("pin_gpu_visible", script, _bundle())
    assert out.patched_text is None
