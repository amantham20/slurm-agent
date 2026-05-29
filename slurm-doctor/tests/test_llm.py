"""Tests for the opt-in LLM fallback (Layer 3).

No network: a fake client returns a canned tool_use response. These verify
secret redaction, payload assembly, response parsing, and that diagnose()
only reaches for the LLM when nothing else explained the failure.
"""
from __future__ import annotations

from pathlib import Path

from slurm_doctor.collect import CollectedBundle, StdioFile
from slurm_doctor.diagnose import diagnose
from slurm_doctor.llm import LLMResult, build_payload, query_llm, redact
from slurm_doctor.parse import load_rules


# -- redaction --------------------------------------------------------------

def test_redact_masks_secret_assignments():
    text = (
        "AWS_SECRET_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE\n"
        "MY_API_TOKEN: abcdef123456\n"
        "DB_PASSWORD=hunter2\n"
        "HARMLESS=keepme\n"
    )
    out = redact(text)
    assert "AKIAIOSFODNN7EXAMPLE" not in out
    assert "abcdef123456" not in out
    assert "hunter2" not in out
    assert "keepme" in out  # non-secret survives
    assert out.count("[REDACTED]") == 3


def test_redact_masks_known_token_shapes():
    text = "using sk-ant1234567890abcdef and ghp_0123456789abcdefghijABCD here"
    out = redact(text)
    assert "sk-ant1234567890abcdef" not in out
    assert "ghp_0123456789abcdefghijABCD" not in out
    assert "[REDACTED]" in out


def test_redact_handles_empty():
    assert redact("") == ""


# -- payload ----------------------------------------------------------------

def test_build_payload_tails_and_redacts(tmp_path):
    err = tmp_path / "stderr.txt"
    err.write_text("\n".join(f"line {i}" for i in range(500)) + "\nAWS_KEY=topsecret\n")
    b = CollectedBundle(jobid="1", cache_dir=str(tmp_path), state="FAILED",
                        exit_code="1:0", req_mem="1G")
    b.stderr = StdioFile(declared_path="x", cached_path=str(err))
    payload = build_payload(b)
    assert payload["jobid"] == "1"
    assert payload["state"] == "FAILED"
    # only the tail is included
    assert "line 499" in payload["stderr_tail"]
    assert "line 0" not in payload["stderr_tail"]
    # secrets stripped even in the tail
    assert "topsecret" not in payload["stderr_tail"]


# -- fake client + parsing --------------------------------------------------

class _FakeBlock:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeResp:
    def __init__(self, content):
        self.content = content


class _FakeMessages:
    def __init__(self, captured):
        self._captured = captured

    def create(self, **kwargs):
        self._captured.update(kwargs)
        return _FakeResp([
            _FakeBlock(type="text", text="thinking..."),
            _FakeBlock(
                type="tool_use", name="report_diagnosis",
                input={
                    "category": "environment",
                    "root_cause": "binary not found on PATH",
                    "confidence": 0.7,
                    "contributing_factors": ["module not loaded"],
                    "suggested_fix_kind": "prepend_modules",
                    "evidence_quote": "command not found: foo",
                },
            ),
        ])


class _FakeClient:
    def __init__(self):
        self.captured: dict = {}
        self.messages = _FakeMessages(self.captured)


def _bundle(tmp_path) -> CollectedBundle:
    err = tmp_path / "stderr.txt"
    err.write_text("foo: command not found\n")
    b = CollectedBundle(jobid="9", cache_dir=str(tmp_path), state="FAILED", exit_code="1:0")
    b.stderr = StdioFile(declared_path="x", cached_path=str(err))
    return b


def test_query_llm_parses_tool_use(tmp_path):
    client = _FakeClient()
    res = query_llm(_bundle(tmp_path), client=client)
    assert isinstance(res, LLMResult)
    assert res.category == "environment"
    assert res.confidence == 0.7
    assert res.suggested_fix_kind == "prepend_modules"
    # request was structured: forced the report_diagnosis tool, cached system
    assert client.captured["tool_choice"]["name"] == "report_diagnosis"
    assert client.captured["system"][0]["cache_control"]["type"] == "ephemeral"


def test_query_llm_returns_none_on_no_tool_block(tmp_path):
    class _Empty(_FakeClient):
        def __init__(self):
            super().__init__()
            self.messages = type("M", (), {"create": lambda self, **kw: _FakeResp([_FakeBlock(type="text", text="nope")])})()
    assert query_llm(_bundle(tmp_path), client=_Empty()) is None


# -- diagnose integration ---------------------------------------------------

def test_diagnose_uses_llm_only_when_no_rule_and_inconclusive(tmp_path):
    # stderr with no rule signature, generic FAILED -> LLM should fire
    err = tmp_path / "stderr.txt"
    err.write_text("some opaque application error that matches no rule\n")
    b = CollectedBundle(jobid="9", cache_dir=str(tmp_path), state="FAILED", exit_code="1:0")
    b.stderr = StdioFile(declared_path="x", cached_path=str(err))

    client = _FakeClient()
    d = diagnose(b, load_rules(), use_llm=True, llm_client=client)
    assert d.used_llm is True
    assert d.state_category == "llm:environment"
    assert d.confidence == 0.7
    # LLM fix folded in but gated (never auto-applicable)
    fix = next(f for f in d.proposed_fixes if f.fix_kind == "prepend_modules")
    assert fix.requires_yes is True


def test_diagnose_skips_llm_when_rule_fires(tmp_path):
    # stderr matches missing_executable -> rule wins, LLM never called
    err = tmp_path / "stderr.txt"
    err.write_text("./x: No such file or directory\n")
    b = CollectedBundle(jobid="9", cache_dir=str(tmp_path), state="FAILED", exit_code="127:0")
    b.stderr = StdioFile(declared_path="x", cached_path=str(err))

    client = _FakeClient()
    d = diagnose(b, load_rules(), use_llm=True, llm_client=client)
    assert d.used_llm is False
    assert client.captured == {}  # LLM was never invoked


def test_diagnose_skips_llm_when_state_conclusive(tmp_path):
    # TIMEOUT is conclusive on its own -> no LLM
    b = CollectedBundle(jobid="9", cache_dir=str(tmp_path), state="TIMEOUT",
                        exit_code="0:15", elapsed="00:02:00", timelimit="00:00:30")
    client = _FakeClient()
    d = diagnose(b, load_rules(), use_llm=True, llm_client=client)
    assert d.used_llm is False
    assert client.captured == {}


def test_diagnose_no_llm_by_default(tmp_path):
    err = tmp_path / "stderr.txt"
    err.write_text("opaque error\n")
    b = CollectedBundle(jobid="9", cache_dir=str(tmp_path), state="FAILED", exit_code="1:0")
    b.stderr = StdioFile(declared_path="x", cached_path=str(err))
    client = _FakeClient()
    d = diagnose(b, load_rules(), llm_client=client)  # use_llm defaults False
    assert d.used_llm is False
    assert client.captured == {}
