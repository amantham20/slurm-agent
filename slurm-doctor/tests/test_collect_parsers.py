"""Unit tests for the pure parsers inside collect.py.

These run on the host with no SLURM tools — they only exercise text parsing.
"""
from __future__ import annotations

from conftest import read_fixture
from slurm_doctor.collect import (
    SACCT_COLS,
    _ratio,
    _slice_log_by_time,
    expand_nodelist,
    parse_sacct_parsable2,
    parse_scontrol_kv,
    parse_slurm_time,
)


def test_sacct_parsable2_parses_header_and_steps():
    text = read_fixture("sacct_failed_missing_exec.txt")
    rows = parse_sacct_parsable2(text, SACCT_COLS)
    # header row is dropped, parent + .batch step remain
    assert len(rows) == 2
    parent, batch = rows
    assert parent["JobID"] == "2"
    assert parent["State"] == "FAILED"
    assert parent["ExitCode"] == "127:0"
    assert parent["DerivedExitCode"] == "0:0"
    assert parent["NodeList"] == "c1"
    assert parent["ReqMem"] == "200M"
    assert parent["WorkDir"] == "/data"
    assert batch["JobID"] == "2.batch"
    assert batch["State"] == "FAILED"


def test_sacct_parsable2_handles_empty():
    rows = parse_sacct_parsable2("", SACCT_COLS)
    assert rows == []


def test_sacct_parsable2_handles_no_header():
    # When called with -n the header line isn't there.
    rows = parse_sacct_parsable2("5|5|j|u|p|FAILED|1:0|0:0|None||||||||||||||||||", SACCT_COLS)
    assert len(rows) == 1
    assert rows[0]["JobID"] == "5"
    assert rows[0]["State"] == "FAILED"


def test_scontrol_kv_extracts_key_fields():
    text = read_fixture("scontrol_show_job_failed.txt")
    kv = parse_scontrol_kv(text)
    assert kv["JobId"] == "2"
    assert kv["JobName"] == "sd_demo"
    assert kv["JobState"] == "FAILED"
    assert kv["NodeList"] == "c1"
    assert kv["StdOut"] == "/data/sd_demo_2.out"
    assert kv["StdErr"] == "/data/sd_demo_2.err"
    assert kv["WorkDir"] == "/data"
    assert kv["Command"] == "/data/sd_demo.sh"


def test_scontrol_show_node_parses():
    kv = parse_scontrol_kv(read_fixture("scontrol_show_node_c1.txt"))
    assert kv["NodeName"] == "c1"
    assert "IDLE" in kv["State"]
    assert kv["RealMemory"] == "16080"


def test_expand_nodelist_variants():
    assert expand_nodelist("c1") == ["c1"]
    assert expand_nodelist("c1,c2") == ["c1", "c2"]
    assert expand_nodelist("c[1-3]") == ["c1", "c2", "c3"]
    assert expand_nodelist("c[1,3,5]") == ["c1", "c3", "c5"]
    assert expand_nodelist("c[1-2,5]") == ["c1", "c2", "c5"]
    # mix of bracketed and bare
    assert expand_nodelist("c[1-2],g1") == ["c1", "c2", "g1"]
    assert expand_nodelist("") == []
    # zero-padded preserved when first endpoint starts with 0
    assert expand_nodelist("n[01-03]") == ["n01", "n02", "n03"]


def test_ratio_handles_units():
    # 150M / 200M ~= 0.75
    r = _ratio("150M", "200M")
    assert r is not None and 0.74 < r < 0.76
    # 1.9G vs 2G ~= 0.95
    r = _ratio("1900M", "2G")
    assert r is not None and 0.92 < r < 0.96
    # Garbage in → None
    assert _ratio("garbage", "200M") is None
    assert _ratio("100M", "") is None
    # 'n'/'c' suffix on ReqMem ("4G" per node) is tolerated
    r = _ratio("3500M", "4Gn")
    assert r is not None and 0.85 < r < 0.86


def test_parse_slurm_time():
    t = parse_slurm_time("2026-05-28T02:56:57")
    assert t is not None
    assert t.year == 2026 and t.month == 5 and t.hour == 2
    assert parse_slurm_time("Unknown") is None
    assert parse_slurm_time(None) is None


def test_slice_log_by_jobid_keeps_relevant_lines():
    log = read_fixture("slurmd_log_jid2.txt")
    sliced = _slice_log_by_time(log, start=None, end=None, jobid="2")
    # Without a time window we keep lines mentioning JobId=2 OR step lines
    # tagged [2.batch] / [2.0] / etc.
    assert "JobId=2" in sliced
    assert "task 0 (564) exited with exit code 127" in sliced  # step-tagged
    for line in sliced.splitlines():
        assert "JobId=2" in line or "[2." in line


def test_slice_log_by_window_includes_recent_lines():
    log = read_fixture("slurmd_log_jid2.txt")
    start = parse_slurm_time("2026-05-28T02:56:57")
    end = parse_slurm_time("2026-05-28T02:56:59")
    sliced = _slice_log_by_time(log, start=start, end=end, jobid="2")
    # With a window covering the demo job, the bulk of the log is captured.
    assert "Launching batch JobId=2" in sliced
    assert "task 0 (564) exited with exit code 127" in sliced
