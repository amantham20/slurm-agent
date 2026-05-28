"""Tests for the layered classifier."""
from __future__ import annotations

from pathlib import Path

import pytest

from conftest import FIXTURES
from slurm_doctor.collect import CollectedBundle, StdioFile
from slurm_doctor.parse import (
    Rule,
    RuleParseError,
    apply_rules,
    classify_state,
    load_rules,
)


# -- Layer 1 state machine ---------------------------------------------------

def _bundle(**kw) -> CollectedBundle:
    b = CollectedBundle(jobid="X", cache_dir="/tmp/x")
    for k, v in kw.items():
        setattr(b, k, v)
    return b


def test_state_timeout():
    s = classify_state(_bundle(state="TIMEOUT", exit_code="0:15",
                               elapsed="00:01:00", timelimit="00:00:30"))
    assert s.category == "timeout"
    assert s.confidence == 1.0


def test_state_oom():
    s = classify_state(_bundle(state="OUT_OF_MEMORY", exit_code="0:9",
                               req_mem="1G", max_rss="1.1G"))
    assert s.category == "oom"


def test_state_node_fail():
    s = classify_state(_bundle(state="NODE_FAIL", nodelist="c1", reason="NonResp"))
    assert s.category == "node_fail"


def test_state_cancelled_user_vs_signalled():
    user = classify_state(_bundle(state="CANCELLED", exit_code="0:0"))
    assert user.category == "cancelled_user"

    signalled = classify_state(_bundle(state="CANCELLED", exit_code="0:15"))
    assert signalled.category == "cancelled_signalled"


def test_state_failed_generic():
    s = classify_state(_bundle(state="FAILED", exit_code="127:0"))
    assert s.category == "failed_generic"


def test_state_completed():
    s = classify_state(_bundle(state="COMPLETED", exit_code="0:0"))
    assert s.category == "completed"


def test_state_unknown():
    s = classify_state(_bundle(state="", exit_code=""))
    assert s.category == "unknown"


# -- Layer 2 rule pack -------------------------------------------------------

def test_builtin_rules_load_and_validate():
    rules = load_rules()
    assert len(rules) >= 6
    ids = {r.id for r in rules}
    # Spot-check a few canonical ones
    for must in ("missing_executable", "missing_module", "walltime_exceeded",
                 "python_module_not_found", "slurm_oom_state", "segfault"):
        assert must in ids


def test_rule_validation_rejects_unknown_predicate(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "- id: bogus\n  category: x\n  match: {unknown_thing: foo}\n  hint: nope\n"
    )
    with pytest.raises(RuleParseError):
        load_rules(tmp_path)


def test_rule_validation_rejects_missing_keys(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- id: x\n  category: c\n")  # no match, no hint
    with pytest.raises(RuleParseError):
        load_rules(tmp_path)


# -- end-to-end: real fixture against real rules ----------------------------

def _bundle_pointing_at_fixtures(tmp_path: Path) -> CollectedBundle:
    """Build a bundle whose stderr.cached_path points at the real fixture."""
    stderr_file = tmp_path / "stderr.txt"
    stderr_file.write_text((FIXTURES / "stderr_missing_exec.txt").read_text())
    b = CollectedBundle(
        jobid="2",
        cache_dir=str(tmp_path),
        state="FAILED",
        exit_code="127:0",
        reason="None",
        nodelist="c1",
        req_mem="200M",
        max_rss=None,
        elapsed="00:00:01",
        timelimit="00:01:00",
        workdir="/data",
    )
    b.stderr = StdioFile(declared_path="/data/sd_demo_2.err",
                         cached_path=str(stderr_file), original_size=104, truncated=False)
    return b


def test_missing_executable_rule_fires_on_real_stderr(tmp_path):
    bundle = _bundle_pointing_at_fixtures(tmp_path)
    hits = apply_rules(bundle, load_rules())
    rule_ids = [h.rule_id for h in hits]
    assert "missing_executable" in rule_ids
    me = next(h for h in hits if h.rule_id == "missing_executable")
    assert me.fix_kind == "fix_path"
    # Evidence cites both the ExitCode and a stderr line
    assert any("ExitCode=127:0" in e for e in me.evidence)
    assert any("No such file or directory" in e for e in me.evidence)
    # Line numbers prefixed for traceability
    assert any("stderr line 1:" in e for e in me.evidence)


def test_first_hit_per_category_wins(tmp_path):
    # Two rules in the same category — the first declared one wins.
    rules = [
        Rule(id="first", title="first", category="dup", match={"stderr_regex": "."},
             hint="h", fix_kind=None, confidence=0.5),
        Rule(id="second", title="second", category="dup", match={"stderr_regex": "."},
             hint="h", fix_kind=None, confidence=0.9),
    ]
    bundle = _bundle_pointing_at_fixtures(tmp_path)
    hits = apply_rules(bundle, rules)
    assert len(hits) == 1
    assert hits[0].rule_id == "first"
