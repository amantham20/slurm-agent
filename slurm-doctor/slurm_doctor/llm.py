"""Layer 3: optional LLM fallback via the Anthropic SDK.

Opt-in ONLY (``--llm`` / ``SLURM_DOCTOR_LLM=1``); never invoked by default. Used
when the rule engine produced no hit and the SLURM state machine wasn't
conclusive. Sends a compact, secret-redacted bundle and asks for a strict
structured verdict via tool use (works across every tool-capable Claude model,
including the configured default).

Nothing here imports ``anthropic`` at module load — the import is lazy so the
package works without the SDK installed.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .collect import CollectedBundle

log = logging.getLogger("slurm_doctor.llm")

# The task spec calls for claude-sonnet-4-5; allow override for sites on other models.
DEFAULT_MODEL = os.environ.get("SLURM_DOCTOR_LLM_MODEL", "claude-sonnet-4-5")
MAX_STDERR_LINES = 200
MAX_STDOUT_LINES = 50


# ---------------------------------------------------------------------------
# Secret redaction — never ship anything secret-shaped to the LLM
# ---------------------------------------------------------------------------

# KEY=value / KEY: value where KEY looks like a secret (AWS_, *_TOKEN, *_KEY,
# *PASSWORD*). Case-insensitive; preserves the key, masks the value.
_SECRET_ASSIGN = re.compile(
    r"(?im)^(\s*[\w.\-]*?"
    r"(?:AWS_[A-Z0-9_]*|[A-Z0-9_]*TOKEN|[A-Z0-9_]*KEY|[A-Z0-9_]*PASSWORD[A-Z0-9_]*)"
    r"[\w.\-]*\s*[=:]\s*)(\S+)"
)
# Known token shapes, masked wherever they appear.
_SECRET_TOKEN = re.compile(
    r"\b("
    r"sk-[A-Za-z0-9_\-]{8,}"
    r"|ghp_[A-Za-z0-9]{20,}"
    r"|gho_[A-Za-z0-9]{20,}"
    r"|AKIA[0-9A-Z]{12,}"
    r"|xox[baprs]-[A-Za-z0-9\-]{10,}"
    r"|eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]+"  # JWT
    r")\b"
)

REDACTION = "[REDACTED]"


def redact(text: str) -> str:
    """Mask secret-shaped values. Defensive: applied to everything LLM-bound."""
    if not text:
        return text
    text = _SECRET_ASSIGN.sub(lambda m: m.group(1) + REDACTION, text)
    text = _SECRET_TOKEN.sub(REDACTION, text)
    return text


def _tail(path: str | None, n: int) -> str:
    if not path:
        return ""
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


# ---------------------------------------------------------------------------
# Payload + result
# ---------------------------------------------------------------------------

@dataclass
class LLMResult:
    category: str
    root_cause: str
    confidence: float
    contributing_factors: list[str] = field(default_factory=list)
    suggested_fix_kind: str | None = None
    evidence_quote: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "root_cause": self.root_cause,
            "confidence": self.confidence,
            "contributing_factors": self.contributing_factors,
            "suggested_fix_kind": self.suggested_fix_kind,
            "evidence_quote": self.evidence_quote,
        }


def build_payload(bundle: CollectedBundle) -> dict[str, Any]:
    """Compact, redacted bundle: metadata + tail of stderr/stdout + node state."""
    stderr = redact(_tail(bundle.stderr.cached_path if bundle.stderr else None, MAX_STDERR_LINES))
    stdout = redact(_tail(bundle.stdout.cached_path if bundle.stdout else None, MAX_STDOUT_LINES))
    nodes = {
        n: redact("\n".join(f"{k}={v}" for k, v in (meta.get("state") or {}).items()))
        for n, meta in bundle.nodes.items()
    }
    return {
        "jobid": bundle.jobid,
        "state": bundle.state,
        "exit_code": bundle.exit_code,
        "derived_exit_code": bundle.derived_exit_code,
        "reason": bundle.reason,
        "req_mem": bundle.req_mem,
        "max_rss": bundle.max_rss,
        "elapsed": bundle.elapsed,
        "timelimit": bundle.timelimit,
        "alloc_tres": bundle.alloc_tres,
        "nodelist": bundle.nodelist,
        "stderr_tail": stderr,
        "stdout_tail": stdout,
        "node_state": nodes,
    }


# Strict structured-output contract, expressed as a tool the model must call.
_DIAGNOSIS_TOOL = {
    "name": "report_diagnosis",
    "description": "Report the structured root-cause diagnosis for the failed SLURM job.",
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "Short failure category, e.g. environment, memory, mpi, gpu_driver, disk, unknown.",
            },
            "root_cause": {"type": "string", "description": "One-sentence root cause."},
            "confidence": {"type": "number", "description": "0.0-1.0 confidence."},
            "contributing_factors": {"type": "array", "items": {"type": "string"}},
            "suggested_fix_kind": {
                "type": "string",
                "description": "One of: bump_memory, bump_time, prepend_modules, add_set_eux, "
                               "fix_path, pin_gpu_visible, swap_mpi_launcher, request_constraint, "
                               "add_requeue_guard, or none.",
            },
            "evidence_quote": {
                "type": "string",
                "description": "The single most telling line copied verbatim from the logs.",
            },
        },
        "required": ["category", "root_cause", "confidence"],
    },
}

_SYSTEM = (
    "You are slurm-doctor's fallback analyst. You receive metadata and truncated "
    "logs from a failed SLURM job and must return a single structured diagnosis by "
    "calling the report_diagnosis tool. Be conservative: if the evidence is weak, "
    "use a low confidence and category 'unknown'. Quote a real log line as evidence. "
    "Never invent log content."
)


def query_llm(
    bundle: CollectedBundle,
    *,
    model: str | None = None,
    client: Any | None = None,
    max_tokens: int = 1024,
) -> LLMResult | None:
    """Run the LLM fallback. Returns None (with a logged warning) on any failure
    — a missing SDK, missing key, network error, or malformed response never
    breaks the pipeline."""
    payload = build_payload(bundle)
    if client is None:
        try:
            import anthropic  # lazy: optional dependency
        except ImportError:
            log.warning("llm: anthropic SDK not installed; skipping (pip install 'slurm-doctor[llm]')")
            return None
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            log.warning("llm: no ANTHROPIC_API_KEY in environment; skipping")
            return None
        client = anthropic.Anthropic()

    user_text = (
        "Diagnose this failed SLURM job. Metadata and logs follow as JSON "
        "(logs are already secret-redacted):\n\n"
        + _compact_json(payload)
    )
    try:
        resp = client.messages.create(
            model=model or DEFAULT_MODEL,
            max_tokens=max_tokens,
            system=[{
                "type": "text",
                "text": _SYSTEM,
                # Cache the stable system block — pays off across a sweep of jobs.
                "cache_control": {"type": "ephemeral"},
            }],
            tools=[_DIAGNOSIS_TOOL],
            tool_choice={"type": "tool", "name": "report_diagnosis"},
            messages=[{"role": "user", "content": user_text}],
        )
    except Exception as e:  # noqa: BLE001 - any SDK/network error → graceful skip
        log.warning("llm: request failed: %r", e)
        return None

    return _parse_response(resp)


def _compact_json(payload: dict[str, Any]) -> str:
    import json
    return json.dumps(payload, indent=2, default=str)


def _parse_response(resp: Any) -> LLMResult | None:
    """Pull the report_diagnosis tool_use block out of a Messages response."""
    content = getattr(resp, "content", None) or []
    for block in content:
        btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
        if btype != "tool_use":
            continue
        name = getattr(block, "name", None) or (block.get("name") if isinstance(block, dict) else None)
        if name != "report_diagnosis":
            continue
        data = getattr(block, "input", None)
        if data is None and isinstance(block, dict):
            data = block.get("input")
        if not isinstance(data, dict):
            return None
        fix = data.get("suggested_fix_kind")
        if fix in (None, "none", ""):
            fix = None
        try:
            conf = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        return LLMResult(
            category=str(data.get("category", "unknown")),
            root_cause=str(data.get("root_cause", "")),
            confidence=max(0.0, min(1.0, conf)),
            contributing_factors=list(data.get("contributing_factors", []) or []),
            suggested_fix_kind=fix,
            evidence_quote=data.get("evidence_quote"),
        )
    log.warning("llm: response had no report_diagnosis tool_use block")
    return None
