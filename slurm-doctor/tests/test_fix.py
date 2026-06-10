"""Patch surgery tests: minimal diffs, idempotency, gates."""

from __future__ import annotations

import pytest

from slurm_doctor.diagnose import ProposedFix, diagnose
from slurm_doctor.fix import MAX_CHANGE_RATIO, apply_fix, heal
import slurm_doctor.fix as fix_mod
from slurm_doctor.util import CmdResult

from conftest import load_bundle

SCRIPT = """#!/bin/bash
#SBATCH --job-name=demo
#SBATCH --mem=50M
#SBATCH --time=00:05:00

echo start
python3 big_job.py
"""


def _fix(kind, **params):
    return ProposedFix(kind=kind, title=kind, description="", confidence=0.9, params=params)


def test_bump_memory_minimal_diff(oom_bundle):
    res = apply_fix(_fix("bump_memory", new_mem="448M"), SCRIPT, oom_bundle)
    assert res.changed
    assert "--mem=448M" in res.patched
    assert res.patched.count("\n") == SCRIPT.count("\n")  # value swap only


def test_bump_memory_idempotent(oom_bundle):
    res = apply_fix(_fix("bump_memory", new_mem="448M"), SCRIPT, oom_bundle)
    res2 = apply_fix(_fix("bump_memory", new_mem="448M"), res.patched, oom_bundle)
    assert not res2.changed
    assert any("already set" in n for n in res2.notes)


def test_bump_time_handles_space_separator(oom_bundle):
    script = "#!/bin/bash\n#SBATCH -t 00:05:00\nsleep 1\n"
    res = apply_fix(_fix("bump_time", new_time="00:09:00"), script, oom_bundle)
    assert "-t 00:09:00" in res.patched


def test_bump_inserts_directive_when_absent(oom_bundle):
    script = "#!/bin/bash\n#SBATCH -p cpu\nsleep 1\n"
    res = apply_fix(_fix("bump_memory", new_mem="1G"), script, oom_bundle)
    assert "#SBATCH --mem=1G" in res.patched
    # inserted inside the header, before the body
    assert res.patched.index("--mem=1G") < res.patched.index("sleep 1")


def test_add_set_eux_idempotent(oom_bundle):
    res = apply_fix(_fix("add_set_eux"), SCRIPT, oom_bundle)
    assert "set -euo pipefail" in res.patched
    assert res.patched.index("pipefail") < res.patched.index("echo start")
    res2 = apply_fix(_fix("add_set_eux"), res.patched, oom_bundle)
    assert not res2.changed


def test_add_set_eux_respects_existing_set_e(oom_bundle):
    script = "#!/bin/bash\nset -e\necho hi\n"
    res = apply_fix(_fix("add_set_eux"), script, oom_bundle)
    assert not res.changed


def test_prepend_modules_with_module_names(oom_bundle):
    res = apply_fix(_fix("prepend_modules", modules="gcc openmpi"), SCRIPT, oom_bundle)
    assert "module load gcc" in res.patched and "module load openmpi" in res.patched
    assert "command -v module" in res.patched
    res2 = apply_fix(_fix("prepend_modules", modules="gcc openmpi"), res.patched, oom_bundle)
    assert not res2.changed


def test_swap_mpi_launcher_drops_np(oom_bundle):
    script = "#!/bin/bash\n#SBATCH --ntasks=2\nmpirun -n 8 ./mpi_app --opt\n"
    res = apply_fix(_fix("swap_mpi_launcher"), script, oom_bundle)
    assert "srun --mpi=pmix ./mpi_app --opt" in res.patched
    assert "mpirun" not in res.patched
    assert "-n 8" not in res.patched


def test_fix_path_rewrites_unambiguous_match(oom_bundle, tmp_path, monkeypatch):
    (tmp_path / "bin").mkdir()
    target = tmp_path / "bin" / "solver"
    target.write_text("#!/bin/sh\n")
    monkeypatch.setitem(oom_bundle.parent, "WorkDir", str(tmp_path))
    script = "#!/bin/bash\n/scratch/old/solver --in data\n"
    res = apply_fix(_fix("fix_path", missing_path="/scratch/old/solver"), script, oom_bundle)
    assert res.changed and str(target) in res.patched


def test_fix_path_refuses_ambiguity(oom_bundle, tmp_path, monkeypatch):
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "solver").write_text("")
    monkeypatch.setitem(oom_bundle.parent, "WorkDir", str(tmp_path))
    script = "#!/bin/bash\n./solver\n"
    res = apply_fix(_fix("fix_path", missing_path="./solver"), script, oom_bundle)
    assert not res.changed
    assert any("2 files" in n for n in res.notes)


def test_request_constraint_and_requeue(oom_bundle):
    res = apply_fix(
        _fix("request_constraint", constraint="gpu", partition="gpu", gres="gpu:1"),
        SCRIPT, oom_bundle,
    )
    for needle in ("--constraint=gpu", "--partition=gpu", "--gres=gpu:1"):
        assert needle in res.patched
    res = apply_fix(_fix("add_requeue_guard"), SCRIPT, oom_bundle)
    assert "#SBATCH --requeue" in res.patched
    res2 = apply_fix(_fix("add_requeue_guard"), res.patched, oom_bundle)
    assert not res2.changed


def test_rewrite_guard_downgrades(oom_bundle, tmp_path, monkeypatch):
    """Touching more than 20% of existing lines downgrades to suggest."""
    monkeypatch.setitem(oom_bundle.parent, "WorkDir", str(tmp_path))
    (tmp_path / "x").write_text("")
    # a 3-line script where fix_path would rewrite 2/3 of the body
    script = "/old/x a\n/old/x b\n/old/x c\n"
    res = apply_fix(_fix("fix_path", missing_path="/old/x"), script, oom_bundle)
    assert res.downgraded and not res.applicable
    assert any("downgrading" in n for n in res.notes)


# --------------------------------------------------------------------------
# heal gates (sbatch/squeue mocked)
# --------------------------------------------------------------------------


def _mock_slurm(monkeypatch, squeue_state=""):
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        if argv[0] == "squeue":
            return CmdResult(argv, 0, squeue_state, "", 0.0)
        if argv[0] == "sbatch":
            return CmdResult(argv, 0, "Submitted batch job 99\n", "", 0.0)
        if argv[0] == "sacct":
            return CmdResult(argv, 0, "", "", 0.0)
        return CmdResult(argv, 1, "", "unexpected", 0.0)

    monkeypatch.setattr(fix_mod, "run", fake_run)
    return calls


def test_heal_oom_auto_allowed(oom_bundle, cfg, monkeypatch, tmp_path):
    monkeypatch.setitem(oom_bundle.parent, "WorkDir", str(tmp_path))
    calls = _mock_slurm(monkeypatch)
    diag = diagnose(oom_bundle, cfg)
    out = heal(diag, oom_bundle, cfg, yes=False)
    assert out.healed and out.new_jobid == "99"
    sbatch = next(c for c in calls if c[0] == "sbatch")
    assert any(a.startswith("--comment=slurm-doctor:fix=bump_memory:parent=2") for a in sbatch)
    patched = tmp_path / "oom.fix1.sh"
    assert patched.exists() and "--mem=" in patched.read_text()
    # ledger written
    import json
    chain = json.loads((cfg.report_dir / "chain.json").read_text())
    assert chain["99"]["parent"] == "2"


def test_heal_adds_afternotok_when_parent_pending(oom_bundle, cfg, monkeypatch, tmp_path):
    monkeypatch.setitem(oom_bundle.parent, "WorkDir", str(tmp_path))
    calls = _mock_slurm(monkeypatch, squeue_state="PENDING\n")
    diag = diagnose(oom_bundle, cfg)
    assert heal(diag, oom_bundle, cfg).healed
    sbatch = next(c for c in calls if c[0] == "sbatch")
    assert "--dependency=afternotok:2" in sbatch


def test_heal_risky_fix_requires_yes(cfg, monkeypatch, tmp_path):
    bundle = load_bundle("11")  # mpi -> swap_mpi_launcher (risky class)
    monkeypatch.setitem(bundle.parent, "WorkDir", str(tmp_path))
    calls = _mock_slurm(monkeypatch)
    diag = diagnose(bundle, cfg)
    assert diag.fixes[0].kind == "swap_mpi_launcher"
    out = heal(diag, bundle, cfg, yes=False)
    assert not out.healed and "--yes" in out.reason
    assert not any(c[0] == "sbatch" for c in calls)
    out = heal(diag, bundle, cfg, yes=True)
    assert out.healed


def test_heal_refuses_after_two_heals(oom_bundle, cfg, monkeypatch, tmp_path):
    monkeypatch.setitem(oom_bundle.parent, "WorkDir", str(tmp_path))
    _mock_slurm(monkeypatch)
    import json
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    (cfg.report_dir / "chain.json").write_text(json.dumps({
        "2": {"parent": "1", "fix": "bump_memory"},
        "1": {"parent": "0", "fix": "bump_memory"},
    }))
    diag = diagnose(oom_bundle, cfg)
    out = heal(diag, oom_bundle, cfg, yes=True)
    assert not out.healed and "heal(s) deep" in out.reason


def test_heal_comment_chain_depth(oom_bundle, cfg, monkeypatch, tmp_path):
    """Chain depth also derived from the durable sacct Comment trail."""
    monkeypatch.setitem(
        oom_bundle.parent, "Comment", "slurm-doctor:fix=bump_memory:parent=1"
    )
    monkeypatch.setitem(oom_bundle.parent, "WorkDir", str(tmp_path))

    def fake_run(argv, **kw):
        if argv[0] == "sacct":
            return CmdResult(argv, 0, "slurm-doctor:fix=bump_time:parent=0\n", "", 0.0)
        if argv[0] == "squeue":
            return CmdResult(argv, 0, "", "", 0.0)
        if argv[0] == "sbatch":
            return CmdResult(argv, 0, "Submitted batch job 99\n", "", 0.0)
        return CmdResult(argv, 1, "", "", 0.0)

    monkeypatch.setattr(fix_mod, "run", fake_run)
    diag = diagnose(oom_bundle, cfg)
    out = heal(diag, oom_bundle, cfg, yes=True)
    assert not out.healed and "2 heal(s) deep" in out.reason
