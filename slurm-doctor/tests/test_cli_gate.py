"""Tests for the heal-gate and fix-loop-guard helpers in cli.py."""
from __future__ import annotations

from slurm_doctor._shell import ShellResult
from slurm_doctor.cli import _count_chain_healings, _gate_allows
from slurm_doctor.collect import CollectedBundle
from slurm_doctor.diagnose import ProposedFix


def _fix(kind: str, conf: float) -> ProposedFix:
    return ProposedFix(fix_kind=kind, description="d", rationale="r",
                       confidence=conf, requires_yes=False)


def test_gate_blocks_low_confidence():
    safe = {"bump_memory", "bump_time", "add_set_eux"}
    assert not _gate_allows(_fix("bump_memory", 0.80), safe, yes=False)
    assert _gate_allows(_fix("bump_memory", 0.85), safe, yes=False)


def test_gate_blocks_unsafe_kind():
    safe = {"bump_memory", "bump_time", "add_set_eux"}
    assert not _gate_allows(_fix("swap_mpi_launcher", 0.99), safe, yes=False)
    # but --yes overrides
    assert _gate_allows(_fix("swap_mpi_launcher", 0.99), safe, yes=True)


def test_chain_count_walks_parent_chain():
    """Walk: jobid=10 -> parent=8 -> parent=4 (no comment) — count is 2."""
    bundle = CollectedBundle(jobid="10", cache_dir="/tmp")
    responses = {
        "10": "Comment\nslurm-doctor:fix=bump_memory:parent=8\n",
        "8":  "Comment\nslurm-doctor:fix=bump_time:parent=4\n",
        "4":  "Comment\n\n",
    }

    def fake(argv, **kw):
        if argv[:2] == ["sacct", "-j"]:
            jid = argv[2]
            return ShellResult(argv=list(argv), returncode=0,
                               stdout=responses.get(jid, ""), stderr="")
        return ShellResult(argv=list(argv), returncode=1, stdout="", stderr="", missing=True)

    assert _count_chain_healings(bundle, runner=fake) == 2


def test_chain_count_returns_zero_when_no_comment():
    bundle = CollectedBundle(jobid="1", cache_dir="/tmp")

    def fake(argv, **kw):
        return ShellResult(argv=list(argv), returncode=0,
                           stdout="Comment\n\n", stderr="")
    assert _count_chain_healings(bundle, runner=fake) == 0
