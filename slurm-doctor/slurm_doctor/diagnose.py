"""Phase 3: synthesise the SLURM state machine + rule hits into a Diagnosis.

A Diagnosis is what report.py renders. It pins down the root cause, lists
contributing factors with citations, and proposes ranked fixes (each tied to a
fix_kind that Phase 4's fix.py knows how to apply).
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from . import SCHEMA_VERSION
from .collect import CollectedBundle
from .parse import Rule, RuleHit, StateClassification, apply_rules, classify_state

log = logging.getLogger("slurm_doctor.diagnose")


# fix_kind metadata: what the rule writer can pick from + how to describe each
# kind to a human. The actual patcher lives in fix.py (Phase 4).
FIX_KIND_DESCRIPTIONS: dict[str, str] = {
    "bump_memory": "Increase --mem to MaxRSS x 1.3 rounded to the next GB",
    "bump_time": "Increase --time to Elapsed x 1.5, capped at the partition's MaxTime",
    "prepend_modules": "Insert a `module load …` block above the first non-directive line",
    "add_set_eux": "Insert `set -euo pipefail` so silent shell failures stop being silent",
    "fix_path": "Rewrite a missing absolute path if a similar file exists in WorkDir",
    "pin_gpu_visible": "Pin CUDA_VISIBLE_DEVICES / use --gpus-per-task to fix GPU visibility",
    "swap_mpi_launcher": "Replace mpirun/mpiexec with `srun --mpi=pmix`",
    "request_constraint": "Add --constraint=<feature> to avoid the mismatched node",
    "add_requeue_guard": "Add --requeue and a small backoff for transient infrastructure failures",
}


@dataclass
class ProposedFix:
    fix_kind: str
    description: str           # human-readable, always present
    rationale: str             # why this fix follows from the diagnosis
    confidence: float
    requires_yes: bool = True  # gated unless confidence and kind are both safe
    patched_script_path: str | None = None  # filled by fix.py
    diff: str | None = None                 # filled by fix.py


@dataclass
class Diagnosis:
    jobid: str
    schema_version: int = SCHEMA_VERSION
    tldr: str = ""
    state_category: str = "unknown"
    state_headline: str = ""
    state_evidence: list[str] = field(default_factory=list)
    rule_hits: list[dict] = field(default_factory=list)  # serialised RuleHits
    root_cause: str = ""
    contributing_factors: list[str] = field(default_factory=list)
    confidence: float = 0.0
    proposed_fixes: list[ProposedFix] = field(default_factory=list)
    used_llm: bool = False
    bundle_cache_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# fix_kinds considered safe enough to auto-apply via `heal` (confidence>=0.85)
SAFE_FIX_KINDS = {"bump_memory", "bump_time", "add_set_eux"}


def diagnose(
    bundle: CollectedBundle,
    rules: list[Rule],
    *,
    use_llm: bool = False,
    llm_client: Any | None = None,
) -> Diagnosis:
    """Build a Diagnosis from a bundle + rule pack.

    The orchestration is:
      1. Classify the SLURM state (Layer 1).
      2. Apply pattern rules (Layer 2).
      3. Pick a primary cause: state machine wins when conclusive (timeout,
         oom, node_fail, etc.); otherwise the highest-confidence rule hit.
      4. Build ProposedFixes from the primary cause + supporting rules.
      5. Only if no rule fired AND the state wasn't conclusive AND use_llm is
         set, fall back to the LLM (Layer 3).
    """
    state = classify_state(bundle)
    hits = apply_rules(bundle, rules, state=state)

    d = Diagnosis(
        jobid=bundle.jobid,
        state_category=state.category,
        state_headline=state.headline,
        state_evidence=list(state.evidence),
        rule_hits=[_hit_to_dict(h) for h in hits],
        bundle_cache_dir=bundle.cache_dir,
    )

    # Pick the primary explanation.
    primary_state_categories = {
        "timeout", "oom", "node_fail", "boot_fail",
        "preempted", "deadline", "cancelled_user", "cancelled_signalled",
    }
    if state.category in primary_state_categories:
        d.root_cause = state.headline
        d.confidence = state.confidence
        d.contributing_factors = [h.title for h in hits if h.confidence < state.confidence]
        d.tldr = _tldr_for_state(state, bundle)
    elif hits:
        primary = max(hits, key=lambda h: h.confidence)
        d.root_cause = primary.title
        d.confidence = primary.confidence
        d.contributing_factors = [h.title for h in hits if h is not primary]
        d.tldr = _tldr_for_rule(primary, bundle, state)
    else:
        # No state-machine verdict and no rule hits → unknown.
        d.root_cause = state.headline or "Cause unknown — no rules matched"
        d.confidence = state.confidence
        d.tldr = (
            f"Job {bundle.jobid} {state.category} (no rule matched). "
            "Consider rerunning with --llm or adding a rule."
        )

    # Layer 3: LLM fallback, only when nothing else explained the failure.
    llm_res = None
    inconclusive = state.category in ("unknown", "failed_generic", "completed")
    if use_llm and not hits and inconclusive:
        llm_res = _maybe_llm(bundle, llm_client)
        if llm_res is not None:
            d.used_llm = True
            d.root_cause = llm_res.root_cause or d.root_cause
            d.state_category = f"llm:{llm_res.category}"
            d.confidence = llm_res.confidence
            d.tldr = f"Job {bundle.jobid} (LLM): {llm_res.root_cause}"
            d.contributing_factors = list(llm_res.contributing_factors)
            if llm_res.evidence_quote:
                d.state_evidence.append(f"LLM evidence: {llm_res.evidence_quote}")

    # Build ProposedFixes. State-machine causes have canonical fixes; rule
    # hits with fix_kinds add more options.
    d.proposed_fixes = _build_fixes(state, hits, bundle)

    # Fold an LLM-suggested fix in (always gated — LLM fixes are never
    # auto-applicable, so confidence is capped below the heal threshold).
    if llm_res is not None and llm_res.suggested_fix_kind in FIX_KIND_DESCRIPTIONS \
            and llm_res.suggested_fix_kind not in {f.fix_kind for f in d.proposed_fixes}:
        kind = llm_res.suggested_fix_kind
        d.proposed_fixes.insert(0, ProposedFix(
            fix_kind=kind,
            description=FIX_KIND_DESCRIPTIONS[kind],
            rationale="Suggested by the LLM fallback (review before applying)",
            confidence=min(d.confidence, 0.84),
            requires_yes=True,
        ))
    return d


def _maybe_llm(bundle: CollectedBundle, llm_client: Any | None):
    """Import + call the LLM layer lazily so the SDK stays an optional dep."""
    try:
        from .llm import query_llm
    except Exception as e:  # noqa: BLE001
        log.warning("llm import failed: %r", e)
        return None
    res = query_llm(bundle, client=llm_client)
    return res


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hit_to_dict(h: RuleHit) -> dict:
    return {
        "rule_id": h.rule_id,
        "title": h.title,
        "category": h.category,
        "hint": h.hint,
        "fix_kind": h.fix_kind,
        "confidence": h.confidence,
        "evidence": list(h.evidence),
    }


def _tldr_for_state(state: StateClassification, bundle: CollectedBundle) -> str:
    if state.category == "timeout":
        return (
            f"Job {bundle.jobid} hit walltime "
            f"(Timelimit={bundle.timelimit}, Elapsed={bundle.elapsed})."
        )
    if state.category == "oom":
        return (
            f"Job {bundle.jobid} was OOM-killed by SLURM "
            f"(ReqMem={bundle.req_mem}, MaxRSS={bundle.max_rss})."
        )
    if state.category == "node_fail":
        return f"Job {bundle.jobid} failed because node {bundle.nodelist} went down mid-job."
    if state.category == "cancelled_user":
        return f"Job {bundle.jobid} was cancelled cleanly — likely scancel by the user."
    if state.category == "cancelled_signalled":
        return f"Job {bundle.jobid} was killed via signal (ExitCode={bundle.exit_code})."
    if state.category == "preempted":
        return f"Job {bundle.jobid} was preempted by the scheduler."
    return f"Job {bundle.jobid} {state.category}: {state.headline}."


def _tldr_for_rule(hit: RuleHit, bundle: CollectedBundle, state: StateClassification) -> str:
    return f"Job {bundle.jobid} failed: {hit.title.lower()} (rule {hit.rule_id}, conf {hit.confidence:.2f})."


def _build_fixes(
    state: StateClassification,
    hits: list[RuleHit],
    bundle: CollectedBundle,
) -> list[ProposedFix]:
    fixes: list[ProposedFix] = []
    seen_kinds: set[str] = set()

    def _push(kind: str | None, confidence: float, rationale: str) -> None:
        if not kind or kind in seen_kinds:
            return
        seen_kinds.add(kind)
        fixes.append(
            ProposedFix(
                fix_kind=kind,
                description=FIX_KIND_DESCRIPTIONS.get(kind, kind),
                rationale=rationale,
                confidence=confidence,
                requires_yes=not (confidence >= 0.85 and kind in SAFE_FIX_KINDS),
            )
        )

    # State-machine-driven fix.
    if state.category == "timeout":
        _push("bump_time", state.confidence,
              f"Elapsed={bundle.elapsed} >= Timelimit={bundle.timelimit}")
    elif state.category == "oom":
        _push("bump_memory", state.confidence,
              f"State=OUT_OF_MEMORY with MaxRSS={bundle.max_rss}")
    elif state.category == "node_fail":
        _push("add_requeue_guard", state.confidence, "Transient node failure can be auto-requeued")
        _push("request_constraint", 0.6, "Pinning a feature can avoid the bad node")
    elif state.category == "cancelled_signalled" and bundle.exit_code != "0:15":
        # Probably an admin signal — no automated fix is safe.
        pass

    # Rule-driven fixes.
    for h in hits:
        _push(h.fix_kind, h.confidence, f"Rule {h.rule_id} matched (conf {h.confidence:.2f})")

    # `add_set_eux` is always a safe diagnostic hardening when the script
    # didn't already use it. Keep it last so it shows up as a follow-up.
    if not _script_uses_set_eux(bundle.submit_script_path):
        _push(
            "add_set_eux",
            0.85,
            "Script doesn't `set -euo pipefail`; silent failures will keep slipping through.",
        )

    return fixes


def _script_uses_set_eux(path: str | None) -> bool:
    if not path:
        return False
    try:
        from pathlib import Path
        text = Path(path).read_text(errors="replace")
    except OSError:
        return False
    for line in text.splitlines()[:30]:
        stripped = line.strip()
        if stripped.startswith("set") and ("-e" in stripped or "errexit" in stripped):
            return True
    return False
