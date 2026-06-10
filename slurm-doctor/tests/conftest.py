"""Shared test fixtures: fixture-backed bundles, no cluster required."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slurm_doctor.collect import Bundle
from slurm_doctor.config import Config

FIXTURES = Path(__file__).parent / "fixtures"


def load_bundle(jobid: str) -> Bundle:
    d = FIXTURES / f"job{jobid}"
    manifest = json.loads((d / "manifest.json").read_text())
    return Bundle(jobid, d, manifest)


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(cache_dir=tmp_path / "cache", report_dir=tmp_path / "reports")


@pytest.fixture
def oom_bundle() -> Bundle:
    return load_bundle("2")


@pytest.fixture
def timeout_bundle() -> Bundle:
    return load_bundle("3")
