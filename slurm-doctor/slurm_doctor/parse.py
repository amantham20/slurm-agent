"""Layered failure classifier — Phase 2 of slurm-doctor.

Layer 1 is a tiny state machine over (State × ExitCode × Reason). Layer 2 loads
declarative pattern rules from ``rules/*.yaml`` and walks them against the
collected bundle (stderr/stdout/slurmd.log/dmesg + accounting fields).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from .collect import CollectedBundle, _ratio

log = logging.getLogger("slurm_doctor.parse")


# ---------------------------------------------------------------------------
# Layer 1: state machine
# ---------------------------------------------------------------------------

@dataclass
class StateClassification:
    category: str         # canonical: timeout, oom, node_fail, cancelled_user, ...
    headline: str         # short human-readable label for the report
    confidence: float     # 1.0 when the SLURM state itself is conclusive
    evidence: list[str] = field(default_factory=list)


# SLURM state strings that warrant attention. We deliberately do NOT collapse
# CANCELLED+0:0 (user-cancelled) with CANCELLED+0:15 (signalled), per the spec.
def classify_state(bundle: CollectedBundle) -> StateClassification:
    state = (bundle.state or "").upper().strip()
    # SLURM occasionally suffixes states with " by NNNNN" (e.g. CANCELLED BY 0).
    base_state = state.split()[0] if state else ""
    exitcode = (bundle.exit_code or "").strip()
    reason = (bundle.reason or "").strip()

    if base_state == "TIMEOUT":
        return StateClassification(
            category="timeout",
            headline="Job exceeded its walltime limit",
            confidence=1.0,
            evidence=[
                f"State=TIMEOUT, Elapsed={bundle.elapsed}, Timelimit={bundle.timelimit}",
            ],
        )

    if base_state in ("OUT_OF_MEMORY", "OOM_KILLED"):
        return StateClassification(
            category="oom",
            headline="Job was killed by SLURM/cgroup for exceeding its memory limit",
            confidence=1.0,
            evidence=[f"State={base_state}, ReqMem={bundle.req_mem}, MaxRSS={bundle.max_rss}"],
        )

    if base_state == "NODE_FAIL":
        return StateClassification(
            category="node_fail",
            headline="Compute node failed during the job",
            confidence=1.0,
            evidence=[f"State=NODE_FAIL, NodeList={bundle.nodelist}, Reason={reason}"],
        )

    if base_state == "BOOT_FAIL":
        return StateClassification(
            category="boot_fail",
            headline="Compute node failed to boot for this job",
            confidence=1.0,
            evidence=[f"State=BOOT_FAIL, NodeList={bundle.nodelist}"],
        )

    if base_state == "PREEMPTED":
        return StateClassification(
            category="preempted",
            headline="Job was preempted by the scheduler",
            confidence=1.0,
            evidence=[f"State=PREEMPTED, Reason={reason}"],
        )

    if base_state == "DEADLINE":
        return StateClassification(
            category="deadline",
            headline="Job hit its deadline before completing",
            confidence=1.0,
            evidence=[f"State=DEADLINE, Reason={reason}"],
        )

    if base_state == "CANCELLED":
        # 0:0  -> clean cancel (user); 0:15 -> signalled (admin/scheduler/timeout)
        if exitcode == "0:0":
            return StateClassification(
                category="cancelled_user",
                headline="Job was cancelled cleanly (no signal)",
                confidence=1.0,
                evidence=[f"State=CANCELLED, ExitCode={exitcode}"],
            )
        return StateClassification(
            category="cancelled_signalled",
            headline=f"Job was cancelled via signal (ExitCode={exitcode})",
            confidence=1.0,
            evidence=[
                f"State=CANCELLED, ExitCode={exitcode}",
                f"Reason={reason}" if reason and reason != "None" else "Reason not recorded",
            ],
        )

    if base_state == "FAILED":
        # FAILED + nonzero exit: rules will narrow it down.
        return StateClassification(
            category="failed_generic",
            headline=f"Job exited non-zero (ExitCode={exitcode})",
            confidence=0.6,
            evidence=[f"State=FAILED, ExitCode={exitcode}, Reason={reason}"],
        )

    if base_state == "COMPLETED":
        # Silent-failure detection happens in the rule layer (stderr matches).
        return StateClassification(
            category="completed",
            headline="Job exited 0 — investigating for silent failure",
            confidence=0.5,
            evidence=[f"State=COMPLETED, ExitCode={exitcode}"],
        )

    # Unknown/missing — leave it to the rules.
    return StateClassification(
        category="unknown",
        headline=f"Unrecognised SLURM state: {state or '(empty)'}",
        confidence=0.0,
        evidence=[f"State={state}, ExitCode={exitcode}"],
    )


# ---------------------------------------------------------------------------
# Layer 2: declarative pattern rules
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    id: str
    title: str
    category: str
    match: dict[str, Any]
    hint: str
    fix_kind: str | None
    confidence: float


@dataclass
class RuleHit:
    rule_id: str
    title: str
    category: str
    hint: str
    fix_kind: str | None
    confidence: float
    evidence: list[str] = field(default_factory=list)


# Predicate set understood by `match:` blocks. Adding new ones means changing
# both load_rules (validation) and _match_rule below.
_KNOWN_PREDICATES = {
    "stderr_regex",
    "stdout_regex",
    "slurmd_log_regex",
    "dmesg_regex",
    "state_in",          # list of canonical state categories from Layer 1
    "exit_code_in",      # list of literal sacct ExitCode strings, e.g. "127:0"
    "reason_regex",
    "max_rss_ratio_above",  # float in [0, 1.5] — MaxRSS/ReqMem threshold
}


class RuleParseError(ValueError):
    pass


def load_rules(rules_dir: str | Path | None = None) -> list[Rule]:
    """Load every YAML file under rules_dir into a flat ordered list."""
    if rules_dir is None:
        rules_dir = Path(__file__).resolve().parent / "rules"
    rules_dir = Path(rules_dir)
    out: list[Rule] = []
    for path in sorted(rules_dir.glob("*.yaml")):
        try:
            docs = yaml.safe_load_all(path.read_text())
            for doc in docs:
                if doc is None:
                    continue
                if not isinstance(doc, list):
                    raise RuleParseError(f"{path.name}: top-level must be a list of rules")
                for raw in doc:
                    out.append(_parse_one(raw, source=path.name))
        except yaml.YAMLError as e:
            raise RuleParseError(f"{path.name}: {e}") from e
    return out


def _parse_one(raw: dict, *, source: str) -> Rule:
    required = ("id", "category", "match", "hint")
    for k in required:
        if k not in raw:
            raise RuleParseError(f"{source}: rule missing required key {k!r}: {raw}")
    match = raw["match"]
    if not isinstance(match, dict) or not match:
        raise RuleParseError(f"{source}: rule {raw['id']!r} has empty/invalid match block")
    unknown = set(match) - _KNOWN_PREDICATES
    if unknown:
        raise RuleParseError(
            f"{source}: rule {raw['id']!r} uses unknown predicates {sorted(unknown)}"
        )
    return Rule(
        id=str(raw["id"]),
        title=str(raw.get("title") or raw["id"]),
        category=str(raw["category"]),
        match=match,
        hint=str(raw["hint"]),
        fix_kind=raw.get("fix_kind"),
        confidence=float(raw.get("confidence", 0.7)),
    )


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _read_or_empty(path: str | None) -> str:
    if not path:
        return ""
    try:
        return Path(path).read_text(errors="replace")
    except OSError:
        return ""


def _evidence_for_regex(text: str, pattern: str, *, source_label: str, max_hits: int = 3) -> list[str]:
    """Return up to ``max_hits`` matching lines with 1-based line numbers."""
    rx = re.compile(pattern)
    out: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if rx.search(line):
            out.append(f"{source_label} line {lineno}: {line.strip()}")
            if len(out) >= max_hits:
                break
    return out


def _match_rule(rule: Rule, bundle: CollectedBundle, *, state: StateClassification) -> RuleHit | None:
    """Return a RuleHit iff every predicate in ``rule.match`` matches."""
    ev: list[str] = []

    m = rule.match
    # state_in narrows by Layer-1 category before any text scan.
    if "state_in" in m:
        if state.category not in set(m["state_in"]):
            return None
        ev.append(f"state.category={state.category}")

    # exit_code_in (literal string contains check).
    if "exit_code_in" in m:
        ec = (bundle.exit_code or "").strip()
        if ec not in set(m["exit_code_in"]):
            return None
        ev.append(f"ExitCode={ec}")

    # reason_regex
    if "reason_regex" in m:
        reason = bundle.reason or ""
        if not re.search(m["reason_regex"], reason):
            return None
        ev.append(f"Reason={reason}")

    # max_rss_ratio_above
    if "max_rss_ratio_above" in m:
        ratio = _ratio(bundle.max_rss, bundle.req_mem)
        thr = float(m["max_rss_ratio_above"])
        if ratio is None or ratio < thr:
            return None
        ev.append(f"MaxRSS/ReqMem={ratio:.2f} ≥ {thr}")

    # Text predicates — only read files when a regex is declared.
    text_predicates = [
        ("stderr_regex", bundle.stderr.cached_path if bundle.stderr else None, "stderr"),
        ("stdout_regex", bundle.stdout.cached_path if bundle.stdout else None, "stdout"),
        ("dmesg_regex", bundle.dmesg_path, "dmesg"),
    ]
    for key, path, label in text_predicates:
        if key not in m:
            continue
        text = _read_or_empty(path)
        hits = _evidence_for_regex(text, m[key], source_label=label)
        if not hits:
            return None
        ev.extend(hits)

    # slurmd_log_regex spans all collected nodes.
    if "slurmd_log_regex" in m:
        hit_any = False
        for node, meta in bundle.nodes.items():
            text = _read_or_empty(meta.get("slurmd_log_path"))
            if not text:
                continue
            hits = _evidence_for_regex(
                text, m["slurmd_log_regex"], source_label=f"slurmd@{node}"
            )
            if hits:
                hit_any = True
                ev.extend(hits)
        if not hit_any:
            return None

    return RuleHit(
        rule_id=rule.id,
        title=rule.title,
        category=rule.category,
        hint=rule.hint,
        fix_kind=rule.fix_kind,
        confidence=rule.confidence,
        evidence=ev,
    )


def apply_rules(
    bundle: CollectedBundle,
    rules: list[Rule],
    *,
    state: StateClassification | None = None,
) -> list[RuleHit]:
    """Run every rule against the bundle, returning hits in declaration order.

    Same-category rules: first hit per category wins (kept in order).
    """
    if state is None:
        state = classify_state(bundle)
    hits: list[RuleHit] = []
    seen_categories: set[str] = set()
    for rule in rules:
        if rule.category in seen_categories:
            continue
        h = _match_rule(rule, bundle, state=state)
        if h is None:
            continue
        hits.append(h)
        seen_categories.add(h.category)
    return hits
