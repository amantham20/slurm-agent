"""Phase 3 — Diagnose: combine the state machine, the rule engine and the
(opt-in) LLM fallback into a Diagnosis with ranked ProposedFixes."""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import re

from .collect import Bundle
from .config import Config, RISKY_FIX_KINDS
from .parse import Evidence, RuleEngine, RuleHit, StateClass, classify_state
from .util import (
    format_mem_mb,
    format_timelimit,
    parse_elapsed_to_seconds,
    parse_mem_to_bytes,
    redact,
)

log = logging.getLogger("slurm_doctor.diagnose")

DIAGNOSIS_SCHEMA_VERSION = 1


@dataclasses.dataclass
class ProposedFix:
    kind: str
    title: str
    description: str
    confidence: float
    params: dict = dataclasses.field(default_factory=dict)
    requires_yes: bool = False
    source_rule: str | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Diagnosis:
    jobid: str
    category: str
    root_cause: str
    detail: str
    confidence: float
    state: StateClass
    evidence: list[Evidence]
    contributing: list[str]
    fixes: list[ProposedFix]
    rule_ids: list[str]
    used_llm: bool = False

    def tldr(self) -> str:
        job = self.jobid
        fix = f" Suggested fix: {self.fixes[0].title}." if self.fixes else ""
        return f"Job {job}: {self.root_cause}{fix}"

    def to_dict(self) -> dict:
        return {
            "schema_version": DIAGNOSIS_SCHEMA_VERSION,
            "jobid": self.jobid,
            "category": self.category,
            "root_cause": self.root_cause,
            "detail": self.detail,
            "confidence": self.confidence,
            "state": dataclasses.asdict(self.state),
            "evidence": [dataclasses.asdict(e) for e in self.evidence],
            "contributing": self.contributing,
            "fixes": [f.to_dict() for f in self.fixes],
            "rule_ids": self.rule_ids,
            "used_llm": self.used_llm,
        }


# --------------------------------------------------------------------------
# Fix parameter computation
# --------------------------------------------------------------------------


def _partition_max_time_seconds(bundle: Bundle) -> int | None:
    info = bundle.partition_info or ""
    m = re.search(r"MaxTime=(\S+)", info)
    if not m or m.group(1) in ("UNLIMITED", "INFINITE"):
        return None
    return parse_elapsed_to_seconds(m.group(1))


def build_fix(kind: str, bundle: Bundle, hit: RuleHit | None, base_conf: float) -> ProposedFix | None:
    """Compute concrete parameters for a fix kind from the bundle."""
    parent = bundle.parent
    params: dict = dict(hit.rule.fix_params) if hit else {}
    if hit:
        params.update(hit.captures)
    conf = base_conf

    if kind == "bump_memory":
        max_rss = bundle.max_rss_bytes()
        req = parse_mem_to_bytes(parent.get("ReqMem"))
        if max_rss:
            target = format_mem_mb(math.ceil(max_rss * 1.3))
        elif req:
            target = format_mem_mb(req * 2)
            conf = min(conf, 0.7)  # no MaxRSS evidence; doubling is a guess
        else:
            return None
        params |= {"new_mem": target, "max_rss_bytes": max_rss,
                   "req_mem": parent.get("ReqMem")}
        title = f"raise memory request to --mem={target}"
        desc = (
            f"Peak measured RSS was "
            f"{format_mem_mb(max_rss) if max_rss else 'unknown'} against a request of "
            f"{parent.get('ReqMem') or 'unknown'}; resubmit with --mem={target} "
            f"(peak x 1.3, rounded up)."
        )
    elif kind == "bump_time":
        elapsed = parse_elapsed_to_seconds(parent.get("Elapsed"))
        limit = parse_elapsed_to_seconds(parent.get("Timelimit"))
        base = elapsed or limit
        if not base:
            return None
        target_s = math.ceil(base * 1.5 / 60) * 60
        cap = _partition_max_time_seconds(bundle)
        capped = False
        if cap and target_s > cap:
            target_s, capped = cap, True
        target = format_timelimit(target_s)
        params |= {"new_time": target, "elapsed": parent.get("Elapsed"),
                   "old_limit": parent.get("Timelimit"), "capped_by_partition": capped}
        title = f"raise wall time to --time={target}"
        desc = (
            f"The job ran {parent.get('Elapsed')} against a limit of "
            f"{parent.get('Timelimit')}; resubmit with --time={target} "
            f"(elapsed x 1.5{', capped at partition MaxTime' if capped else ''})."
        )
    elif kind == "add_set_eux":
        title = "add `set -euo pipefail` so failures stop the script early"
        desc = ("The script keeps running after a command fails, which hides the "
                "first real error. `set -euo pipefail` makes it stop at the first failure.")
    elif kind == "prepend_modules":
        mods = params.get("modules") or params.get("module_name") or ""
        title = "initialise the module system (and load modules) at the top of the script"
        desc = ("Batch shells do not source /etc/profile.d, so `module` is undefined. "
                "Source the Lmod init script before the first `module load`.")
        params.setdefault("modules", mods)
    elif kind == "fix_path":
        missing = params.get("missing_path", "")
        if missing.startswith(("/bin", "/usr")) or missing in ("module", "ml"):
            return None  # not something we can re-point
        title = f"fix the path of `{missing or 'the missing command'}`"
        desc = (f"`{missing}` does not exist on the compute node. If a file with the "
                "same name exists in the work dir, the patched script points at it; "
                "otherwise install it or load the right environment.")
    elif kind == "pin_gpu_visible":
        title = "fix GPU visibility (srun --gpus-per-task / CUDA_VISIBLE_DEVICES)"
        desc = ("Launch GPU steps with `srun --gpus-per-task` and do not override "
                "CUDA_VISIBLE_DEVICES by hand; SLURM sets it per step.")
    elif kind == "swap_mpi_launcher":
        title = "launch MPI ranks with `srun --mpi=pmix` instead of mpirun/mpiexec"
        desc = ("srun inherits the allocation geometry (-n/--ntasks) and wires PMIx "
                "without a host launcher, which avoids both the missing-mpirun and "
                "rank-mismatch failure modes.")
    elif kind == "request_constraint":
        feat = params.get("constraint", "")
        title = f"request matching node features (--constraint={feat or '<feature>'})"
        desc = ("The job landed on a node without the hardware/features it needs. "
                "Pin it with --constraint" + (f"={feat}" if feat else "=") +
                (f" and --partition={params['partition']}" if params.get("partition") else "") +
                (f" --gres={params['gres']}" if params.get("gres") else "") + ".")
    elif kind == "add_requeue_guard":
        title = "make the job requeue-able (--requeue) for transient failures"
        desc = ("Adds #SBATCH --requeue and --open-mode=append so the scheduler can "
                "rerun the job after node failures or preemption without losing output.")
    else:
        log.warning("unknown fix kind %s", kind)
        return None

    return ProposedFix(
        kind=kind,
        title=title,
        description=desc,
        confidence=round(conf, 3),
        params=params,
        requires_yes=kind in RISKY_FIX_KINDS,
        source_rule=hit.rule.id if hit else None,
    )


# --------------------------------------------------------------------------
# Diagnose
# --------------------------------------------------------------------------


def diagnose(bundle: Bundle, cfg: Config, engine: RuleEngine | None = None) -> Diagnosis:
    engine = engine or RuleEngine.load()
    state = classify_state(bundle.parent)
    hits = engine.evaluate(bundle, state)

    fixes: list[ProposedFix] = []
    evidence: list[Evidence] = []
    rule_ids: list[str] = []
    for hit in hits:
        rule_ids.append(hit.rule.id)
        evidence.extend(hit.evidence)
        if hit.rule.fix_kind:
            fix = build_fix(hit.rule.fix_kind, bundle, hit, hit.rule.confidence)
            if fix and not any(f.kind == fix.kind for f in fixes):
                fixes.append(fix)

    used_llm = False
    if hits:
        top = hits[0] if len(hits) == 1 else max(hits, key=lambda h: h.rule.confidence)
        root_cause = top.rule.hint or state.detail
        category = top.rule.category
        confidence = top.rule.confidence
        contributing = [
            f"[{h.rule.category}/{h.rule.id}] {h.rule.hint}" for h in hits if h is not top
        ]
    else:
        root_cause = state.detail
        category = state.category
        confidence = 0.5 if state.category not in ("unknown",) else 0.3
        contributing = []
        if cfg.llm_enabled:
            llm = _diagnose_llm(bundle, state, cfg)
            if llm:
                used_llm = True
                root_cause = llm["root_cause"]
                category = llm.get("category", category)
                # LLM-derived confidence is capped below the auto-heal gate on
                # purpose: an unverified model guess must never self-apply.
                confidence = min(float(llm.get("confidence", 0.5)), 0.84)
                contributing = llm.get("contributing_factors", [])
                for fk in llm.get("fix_kinds", []):
                    fix = build_fix(fk, bundle, None, confidence)
                    if fix and not any(f.kind == fix.kind for f in fixes):
                        fixes.append(fix)
                for ev in llm.get("evidence_lines", [])[:5]:
                    evidence.append(Evidence("llm-cited", 0, str(ev)[:400]))

    # Opportunistic hygiene fix: a FAILED job whose script has no `set -e`
    # often dies silently downstream of the first real error.
    script = bundle.script or ""
    if (
        state.category == "failed"
        and script
        and not re.search(r"^\s*set\s+-[a-z]*e", script, re.M)
        and not any(f.kind == "add_set_eux" for f in fixes)
    ):
        hygiene = build_fix("add_set_eux", bundle, None, 0.6)
        if hygiene:
            fixes.append(hygiene)

    fixes.sort(key=lambda f: f.confidence, reverse=True)

    if not evidence:
        evidence.append(
            Evidence("sacct", 0,
                     f"State={state.state} ExitCode={bundle.parent.get('ExitCode')} "
                     f"Reason={bundle.parent.get('Reason')}")
        )

    return Diagnosis(
        jobid=bundle.jobid,
        category=category,
        root_cause=root_cause,
        detail=state.detail,
        confidence=confidence,
        state=state,
        evidence=evidence,
        contributing=contributing,
        fixes=fixes,
        rule_ids=rule_ids,
        used_llm=used_llm,
    )


# --------------------------------------------------------------------------
# Layer 3: LLM fallback (opt-in, never default)
# --------------------------------------------------------------------------

_LLM_TOOL = {
    "name": "report_diagnosis",
    "description": "Report the diagnosis of a failed SLURM job.",
    "input_schema": {
        "type": "object",
        "properties": {
            "root_cause": {"type": "string", "description": "One-sentence root cause."},
            "category": {
                "type": "string",
                "enum": ["memory", "time", "disk", "environment", "exec",
                         "application", "gpu", "mpi", "license",
                         "infrastructure", "scheduling", "unknown"],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "contributing_factors": {"type": "array", "items": {"type": "string"}},
            "evidence_lines": {
                "type": "array", "items": {"type": "string"},
                "description": "Verbatim log lines that support the diagnosis.",
            },
            "fix_kinds": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["bump_memory", "bump_time", "prepend_modules",
                             "add_set_eux", "fix_path", "pin_gpu_visible",
                             "swap_mpi_launcher", "request_constraint",
                             "add_requeue_guard"],
                },
            },
        },
        "required": ["root_cause", "category", "confidence", "evidence_lines"],
    },
}


def _tail_lines(text: str | None, n: int) -> str:
    if not text:
        return "(absent)"
    lines = text.splitlines()
    return "\n".join(lines[-n:])


def _diagnose_llm(bundle: Bundle, state: StateClass, cfg: Config) -> dict | None:
    """Send a compact, redacted bundle to Claude and get a structured diagnosis.

    Only called when no rule fired AND the user opted in (--llm /
    SLURM_DOCTOR_LLM=1). Never sends env dumps; all text passes redact().
    """
    try:
        import anthropic
    except ImportError:
        log.warning("--llm requested but the anthropic package is not installed "
                    "(pip install 'slurm-doctor[llm]')")
        return None

    parent = bundle.parent
    meta = {
        k: parent.get(k, "")
        for k in ("JobName", "State", "ExitCode", "DerivedExitCode", "Reason",
                  "Elapsed", "Timelimit", "ReqMem", "MaxRSS", "ReqCPUS",
                  "AllocTRES", "NodeList", "Partition")
    }
    node_status = "\n".join(
        f"--- {n} ---\n" + (bundle.node_info(n) or "")[:1500] for n in bundle.nodes
    )
    prompt = redact(
        "Diagnose this failed SLURM job from the artifacts below. Quote only "
        "lines that actually appear in the logs as evidence.\n\n"
        f"## sacct metadata\n{json.dumps(meta, indent=1)}\n\n"
        f"## submit script\n{(bundle.script or '(absent)')[:4000]}\n\n"
        f"## stderr (last 200 lines)\n{_tail_lines(bundle.stderr_text, 200)}\n\n"
        f"## stdout (last 50 lines)\n{_tail_lines(bundle.stdout_text, 50)}\n\n"
        f"## slurmd log excerpt\n{_tail_lines(bundle.slurmd_log, 50)}\n\n"
        f"## node status\n{node_status[:3000] or '(absent)'}\n"
    )
    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=cfg.llm_model,
            max_tokens=2048,
            tools=[_LLM_TOOL],
            tool_choice={"type": "tool", "name": "report_diagnosis"},
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:  # API/auth/network errors must never crash a run
        log.warning("LLM fallback failed: %s", exc)
        return None
    for block in msg.content:
        if block.type == "tool_use" and block.name == "report_diagnosis":
            log.info("LLM diagnosis used for job %s", bundle.jobid)
            return dict(block.input)
    return None
