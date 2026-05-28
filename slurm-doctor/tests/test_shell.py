"""Tests for the subprocess wrapper.

We exercise the real subprocess path with safe commands (echo, true, false,
sleep) — no SLURM tools needed.
"""
from __future__ import annotations

import pytest

from slurm_doctor._shell import default_runner


def test_run_captures_stdout_and_returncode():
    r = default_runner(["echo", "hello world"])
    assert r.returncode == 0
    assert r.stdout.rstrip() == "hello world"
    assert r.stderr == ""
    assert not r.timed_out
    assert not r.missing


def test_run_nonzero_exit_is_not_an_exception():
    r = default_runner(["false"])
    assert r.returncode != 0
    assert not r.missing


def test_run_missing_binary_is_flagged_not_raised():
    r = default_runner(["this_binary_does_not_exist_either"])
    assert r.missing is True
    assert r.returncode == 127


def test_run_timeout_returns_partial_output():
    r = default_runner(["sleep", "5"], timeout=0.3)
    assert r.timed_out is True


def test_run_rejects_empty_argv():
    with pytest.raises(ValueError):
        default_runner([])
