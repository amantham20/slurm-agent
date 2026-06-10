"""Rule engine and state machine tests, including mini-YAML/PyYAML parity."""

from __future__ import annotations

from pathlib import Path

import pytest

from slurm_doctor import _yaml
from slurm_doctor.parse import RULES_DIR, RuleEngine, classify_state

from conftest import load_bundle


# --------------------------------------------------------------------------
# mini-yaml parity: every shipped rule file must parse identically
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", sorted(RULES_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_mini_yaml_matches_pyyaml(path: Path, monkeypatch):
    yaml = pytest.importorskip("yaml")
    text = path.read_text()
    expected = yaml.safe_load(text)
    monkeypatch.setattr(_yaml, "_pyyaml", None)  # force the fallback parser
    assert _yaml.safe_load(text) == expected


def test_mini_yaml_scalars(monkeypatch):
    monkeypatch.setattr(_yaml, "_pyyaml", None)
    out = _yaml.safe_load(
        "- id: x\n"
        "  num: 42\n"
        "  f: 0.9\n"
        "  flag: true\n"
        "  s: 'it''s quoted'\n"
        "  lst: [FAILED, TIMEOUT]\n"
        "  match:\n"
        "    stderr_regex: 'a|b'\n"
    )
    assert out == [{
        "id": "x", "num": 42, "f": 0.9, "flag": True, "s": "it's quoted",
        "lst": ["FAILED", "TIMEOUT"], "match": {"stderr_regex": "a|b"},
    }]


# --------------------------------------------------------------------------
# Layer 1: state machine
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state,exitcode,expect_cat,expect_by",
    [
        ("FAILED", "127:0", "failed", None),
        ("TIMEOUT", "0:0", "timeout", None),
        ("OUT_OF_MEMORY", "0:125", "oom", None),
        ("NODE_FAIL", "0:0", "node_fail", None),
        ("BOOT_FAIL", "0:0", "boot_fail", None),
        ("DEADLINE", "0:0", "deadline", None),
        ("PREEMPTED", "0:15", "preempted", None),
        ("CANCELLED by 1000", "0:0", "cancelled", "user"),
        ("CANCELLED by 0", "0:0", "cancelled", "admin"),
        ("COMPLETED", "0:0", "success", None),
    ],
)
def test_classify_state(state, exitcode, expect_cat, expect_by):
    sc = classify_state({"State": state, "ExitCode": exitcode, "User": "alice"})
    assert sc.category == expect_cat
    assert sc.cancelled_by == expect_by


def test_cancelled_signal_15_is_not_user_cancel():
    """CANCELLED+0:0 (user scancel) must not be conflated with 0:15."""
    plain = classify_state({"State": "CANCELLED by 1000", "ExitCode": "0:0"})
    signalled = classify_state({"State": "CANCELLED", "ExitCode": "0:15"})
    assert plain.cancelled_by == "user" and plain.exit_signal in (0, None)
    assert signalled.exit_signal == 15
    assert "signal 15" in signalled.detail


def test_failed_via_signal_11():
    sc = classify_state({"State": "FAILED", "ExitCode": "0:11"})
    assert sc.exit_signal == 11 and "signal 11" in sc.detail


# --------------------------------------------------------------------------
# Layer 2: rules against real captured bundles
# --------------------------------------------------------------------------

ENGINE = RuleEngine.load()

EXPECTED = {
    # jobid: (top-level rule that must fire, category)
    "2": ("oom_stderr_only", "memory"),
    "3": ("walltime_exceeded_with_log", "time"),
    "5": ("missing_executable", "exec"),
    "6": ("python_module_not_found", "application"),
    "8": ("missing_module", "environment"),
    "9": ("segfault_signal", "application"),
    "10": ("disk_full", "disk"),
    "11": ("mpirun_not_found", "mpi"),
    "13": ("cuda_on_cpu_node", "gpu"),
}


@pytest.mark.parametrize("jobid", sorted(EXPECTED), ids=lambda j: f"job{j}")
def test_rules_fire_on_real_bundles(jobid):
    bundle = load_bundle(jobid)
    hits = ENGINE.evaluate(bundle)
    rule_ids = [h.rule.id for h in hits]
    want_rule, want_cat = EXPECTED[jobid]
    assert want_rule in rule_ids, f"expected {want_rule}, got {rule_ids}"
    by_cat = {h.rule.category: h.rule.id for h in hits}
    assert by_cat[want_cat] == want_rule  # first match wins within category


def test_every_hit_carries_evidence():
    for jobid in EXPECTED:
        for hit in ENGINE.evaluate(load_bundle(jobid)):
            assert hit.evidence, f"{hit.rule.id} fired without evidence on job {jobid}"
            for ev in hit.evidence:
                assert ev.line.strip()


def test_first_match_wins_per_category():
    bundle = load_bundle("2")
    hits = ENGINE.evaluate(bundle)
    cats = [h.rule.category for h in hits]
    assert len(cats) == len(set(cats))


def test_rule_files_all_load():
    # every shipped rule compiled (bad regexes are skipped with a warning,
    # which this guards against)
    ids = [r.id for r in ENGINE.rules]
    assert len(ids) == len(set(ids)), "duplicate rule ids"
    assert len(ids) >= 25
    for required in (
        "missing_module", "missing_executable", "permission_denied",
        "mpi_launch_failed", "cuda_driver_mismatch", "gpu_oom",
        "oom_killed", "walltime_exceeded", "disk_quota", "stale_mount",
        "python_module_not_found", "segfault_stderr", "license_unreachable",
        "node_drained_midjob",
    ):
        assert required in ids, f"required rule {required} missing"
