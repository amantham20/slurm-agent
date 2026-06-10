"""Phase 2 — Parse: layered classification of a collected bundle.

Layer 1: the SLURM state machine (State x ExitCode x Reason).
Layer 2: declarative pattern rules from ``rules/*.yaml``.
Layer 3 (LLM fallback) lives in :mod:`slurm_doctor.diagnose`.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from pathlib import Path

from . import _yaml
from .collect import Bundle

log = logging.getLogger("slurm_doctor.parse")

RULES_DIR = Path(__file__).parent / "rules"


# --------------------------------------------------------------------------
# Layer 1: SLURM state machine
# --------------------------------------------------------------------------


@dataclasses.dataclass
class StateClass:
    """Coarse classification from State x ExitCode x Reason."""

    state: str               # raw sacct State, e.g. "CANCELLED by 0"
    category: str            # failed|timeout|oom|node_fail|cancelled|...
    detail: str              # human phrasing of the state
    exit_rc: int | None      # left side of ExitCode
    exit_signal: int | None  # right side of ExitCode
    cancelled_by: str | None = None  # user|admin|scheduler|None
    reason: str | None = None


_STATE_MAP = {
    "FAILED": ("failed", "the job's processes exited with a non-zero status"),
    "TIMEOUT": ("timeout", "the job hit its --time wall clock limit and was killed"),
    "OUT_OF_MEMORY": ("oom", "the job exceeded its memory request and was killed"),
    "NODE_FAIL": ("node_fail", "a node allocated to the job failed mid-run"),
    "BOOT_FAIL": ("boot_fail", "an allocated node failed to boot/provision"),
    "DEADLINE": ("deadline", "the job missed its --deadline and was terminated"),
    "PREEMPTED": ("preempted", "the job was preempted by higher-priority work"),
    "CANCELLED": ("cancelled", "the job was cancelled before completing"),
    "REVOKED": ("cancelled", "the job's sibling allocation was revoked"),
    "COMPLETED": ("success", "the job completed with exit code 0"),
    "RUNNING": ("running", "the job is still running"),
    "PENDING": ("pending", "the job has not started yet"),
}


def parse_exit_code(exitcode: str | None) -> tuple[int | None, int | None]:
    """sacct ExitCode is ``rc:signal``."""
    if not exitcode or ":" not in exitcode:
        return None, None
    rc_s, sig_s = exitcode.split(":", 1)
    try:
        return int(rc_s), int(sig_s)
    except ValueError:
        return None, None


def classify_state(parent: dict) -> StateClass:
    raw_state = (parent.get("State") or "UNKNOWN").strip()
    base = raw_state.split()[0].split("+")[0]
    rc, sig = parse_exit_code(parent.get("ExitCode"))
    drc, dsig = parse_exit_code(parent.get("DerivedExitCode"))
    category, detail = _STATE_MAP.get(base, ("unknown", f"unrecognised state {base}"))

    cancelled_by = None
    if base in ("CANCELLED", "REVOKED"):
        m = re.search(r"by\s+(\S+)", raw_state)
        if m:
            who = m.group(1)
            if who in ("0", "root", "slurm"):
                cancelled_by = "admin"
            else:
                cancelled_by = "user"
        reason = (parent.get("Reason") or "").lower()
        if "preempt" in reason or base == "REVOKED":
            cancelled_by = "scheduler"
        if cancelled_by == "user" and parent.get("User") and m:
            # uid matching the submitting user => self-cancel
            cancelled_by = "user"
        if sig and sig not in (0,):
            detail = f"the job was cancelled with signal {sig}"
            if cancelled_by is None:
                cancelled_by = "scheduler"
        elif cancelled_by:
            detail = f"the job was cancelled by the {cancelled_by}"
    # FAILED via signal (e.g. 0:11 segfault, 0:9 oom-ish kill)
    if base == "FAILED" and sig:
        detail = f"the job was killed by signal {sig}"

    return StateClass(
        state=raw_state,
        category=category,
        detail=detail,
        exit_rc=rc if rc is not None else drc,
        exit_signal=sig if sig else dsig,
        cancelled_by=cancelled_by,
        reason=parent.get("Reason") or None,
    )


# --------------------------------------------------------------------------
# Layer 2: declarative rules
# --------------------------------------------------------------------------

MATCH_TEXT_KEYS = (
    "stderr_regex",
    "stdout_regex",
    "script_regex",
    "slurmd_regex",
    "dmesg_regex",
    "sacct_regex",
    "jobcomp_regex",
)


@dataclasses.dataclass
class Evidence:
    source: str    # stderr|stdout|script|slurmd|dmesg|sacct|jobcomp|state
    line_no: int   # 1-based; 0 for whole-field evidence
    line: str
    match_count: int = 1

    def cite(self) -> str:
        loc = f"{self.source}:{self.line_no}" if self.line_no else self.source
        return f"`{loc}`: `{self.line.strip()}`"


@dataclasses.dataclass
class Rule:
    id: str
    category: str
    match: dict
    hint: str = ""
    fix_kind: str | None = None
    fix_params: dict = dataclasses.field(default_factory=dict)
    confidence: float = 0.5
    priority: int = 50
    source_file: str = ""

    def __post_init__(self):
        self._compiled = {
            k: re.compile(v, re.M) for k, v in self.match.items() if k in MATCH_TEXT_KEYS
        }


@dataclasses.dataclass
class RuleHit:
    rule: Rule
    evidence: list[Evidence]
    captures: dict[str, str] = dataclasses.field(default_factory=dict)


def _first_match_evidence(source: str, text: str, rx: re.Pattern) -> tuple[Evidence, dict] | None:
    matches = list(rx.finditer(text))
    if not matches:
        return None
    m = matches[0]
    line_no = text.count("\n", 0, m.start()) + 1
    line_start = text.rfind("\n", 0, m.start()) + 1
    line_end = text.find("\n", m.start())
    line = text[line_start : line_end if line_end != -1 else len(text)]
    ev = Evidence(source, line_no, line[:400], match_count=len(matches))
    captures = {k: v for k, v in (m.groupdict() or {}).items() if v}
    return ev, captures


class RuleEngine:
    def __init__(self, rules: list[Rule]):
        self.rules = sorted(rules, key=lambda r: r.priority)

    @classmethod
    def load(cls, extra_dirs: list[Path] | None = None) -> "RuleEngine":
        rules: list[Rule] = []
        dirs = [RULES_DIR] + (extra_dirs or [])
        for d in dirs:
            for f in sorted(Path(d).glob("*.yaml")):
                try:
                    data = _yaml.safe_load(f.read_text()) or []
                except Exception as exc:
                    log.warning("skipping unparsable rule file %s: %s", f, exc)
                    continue
                for raw in data:
                    try:
                        rules.append(
                            Rule(
                                id=raw["id"],
                                category=raw["category"],
                                match=raw.get("match") or {},
                                hint=raw.get("hint", ""),
                                fix_kind=raw.get("fix_kind"),
                                fix_params=raw.get("fix_params") or {},
                                confidence=float(raw.get("confidence", 0.5)),
                                priority=int(raw.get("priority", 50)),
                                source_file=f.name,
                            )
                        )
                    except (KeyError, re.error, TypeError, ValueError) as exc:
                        log.warning("skipping bad rule %r in %s: %s", raw, f, exc)
        log.debug("loaded %d rules", len(rules))
        return cls(rules)

    def evaluate(self, bundle: Bundle, state: StateClass | None = None) -> list[RuleHit]:
        """Run all rules. First match wins per category; multiple categories
        may match."""
        state = state or classify_state(bundle.parent)
        sources = bundle.evidence_sources()
        hits: list[RuleHit] = []
        seen_categories: set[str] = set()
        for rule in self.rules:
            if rule.category in seen_categories:
                continue
            hit = self._eval_rule(rule, bundle, state, sources)
            if hit:
                hits.append(hit)
                seen_categories.add(rule.category)
        return hits

    def _eval_rule(
        self, rule: Rule, bundle: Bundle, state: StateClass, sources: dict[str, str]
    ) -> RuleHit | None:
        m = rule.match
        if not m:
            return None
        evidence: list[Evidence] = []
        captures: dict[str, str] = {}

        want_states = m.get("state")
        if want_states:
            if isinstance(want_states, str):
                want_states = [want_states]
            base = state.state.split()[0].split("+")[0]
            if base not in want_states:
                return None
            evidence.append(Evidence("sacct", 0, f"State={state.state}"))
        if "exit_signal" in m:
            if state.exit_signal != int(m["exit_signal"]):
                return None
            evidence.append(
                Evidence("sacct", 0, f"ExitCode={bundle.parent.get('ExitCode')}")
            )
        if "min_exit_code" in m:
            if state.exit_rc is None or state.exit_rc < int(m["min_exit_code"]):
                return None
        if "cancelled_by" in m and state.cancelled_by != m["cancelled_by"]:
            return None
        if "reason_regex" in m:
            if not re.search(m["reason_regex"], state.reason or "", re.I):
                return None
            evidence.append(Evidence("sacct", 0, f"Reason={state.reason}"))

        for key in MATCH_TEXT_KEYS:
            if key not in m:
                continue
            source = key.removesuffix("_regex")
            text = sources.get(source)
            if not text:
                return None
            found = _first_match_evidence(source, text, rule._compiled[key])
            if not found:
                return None
            ev, caps = found
            evidence.append(ev)
            captures.update(caps)

        if not evidence:
            return None
        return RuleHit(rule=rule, evidence=evidence, captures=captures)
