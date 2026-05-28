"""Shared pytest fixtures for slurm-doctor tests."""
from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when running pytest from the repo without installing.
SRC = Path(__file__).resolve().parent.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def read_fixture(name: str) -> str:
    return (FIXTURES / name).read_text()
