from datetime import datetime

import pytest

from slurm_doctor.util import (
    expand_filename_pattern,
    format_mem_mb,
    format_timelimit,
    head_tail_cap,
    parse_elapsed_to_seconds,
    parse_mem_to_bytes,
    parse_since,
    redact,
    run,
)


def test_run_is_list_only():
    with pytest.raises(TypeError):
        run("echo hi")  # type: ignore[arg-type]


def test_run_captures_and_times_out():
    r = run(["sh", "-c", "echo out; echo err >&2; exit 3"])
    assert r.returncode == 3 and r.stdout.strip() == "out" and r.stderr.strip() == "err"
    r = run(["sleep", "5"], timeout=0.2)
    assert r.timed_out and not r.ok


def test_run_missing_binary():
    r = run(["definitely-not-a-real-binary-xyz"])
    assert r.returncode == 127 and "not found" in r.stderr


def test_head_tail_cap_keeps_both_ends():
    text = "\n".join(f"line{i}" for i in range(10000))
    capped, truncated = head_tail_cap(text, 2000)
    assert truncated
    assert "line0" in capped and "line9999" in capped
    assert "truncated" in capped
    assert len(capped.encode()) < 2300


def test_head_tail_cap_passthrough():
    capped, truncated = head_tail_cap("short", 1000)
    assert capped == "short" and not truncated


@pytest.mark.parametrize(
    "value,expected",
    [
        ("50M", 50 * 1024**2),
        ("2G", 2 * 1024**3),
        ("310500K", 310500 * 1024),
        ("1.5G", int(1.5 * 1024**3)),
        ("4000Mn", 4000 * 1024**2),
        ("100", 100 * 1024**2),  # bare numbers are MB
        ("", None),
        (None, None),
    ],
)
def test_parse_mem(value, expected):
    assert parse_mem_to_bytes(value) == expected


def test_format_mem_rounds_sensibly():
    assert format_mem_mb(int(0.394 * 1024**3)) == "448M"  # next 64M step
    assert format_mem_mb(int(1.2 * 1024**3)) == "2G"      # next GB above 1G
    assert format_mem_mb(1024**3) == "1G"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("00:01:26", 86),
        ("1-02:00:00", 93600),
        ("05:00", 300),
        ("UNLIMITED", None),
        (None, None),
    ],
)
def test_parse_elapsed(value, expected):
    assert parse_elapsed_to_seconds(value) == expected


def test_format_timelimit_roundtrip():
    assert format_timelimit(180) == "00:03:00"
    assert format_timelimit(93600) == "1-02:00:00"
    assert parse_elapsed_to_seconds(format_timelimit(86)) >= 86


def test_parse_since_relative():
    now = datetime(2026, 6, 10, 12, 0, 0)
    assert parse_since("1 hour ago", now).hour == 11
    assert parse_since("30 min ago", now).minute == 30
    assert parse_since("now-2hours", now).hour == 10
    assert parse_since("2026-06-10T03:04:05", now).hour == 3
    with pytest.raises(ValueError):
        parse_since("whenever")


def test_redact_masks_secrets():
    text = (
        "export AWS_SECRET_ACCESS_KEY=abc123\n"
        "MY_API_TOKEN=tok\n"
        "DB_PASSWORD=hunter2\n"
        "Authorization: Bearer xyz\n"
        "NORMAL_VAR=ok\n"
    )
    out = redact(text)
    assert "abc123" not in out and "tok" not in out and "hunter2" not in out
    assert "xyz" not in out
    assert "NORMAL_VAR=ok" in out
    assert out.count("[REDACTED]") >= 4


def test_expand_filename_pattern():
    meta = {"jobid": "42", "jobname": "oom", "user": "alice", "node": "c1"}
    assert expand_filename_pattern("/data/%x-%j.out", meta) == "/data/oom-42.out"
    assert expand_filename_pattern("%u/%N/%%.log", meta) == "alice/c1/%.log"
