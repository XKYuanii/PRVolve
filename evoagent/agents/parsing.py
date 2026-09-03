"""Turn a model's raw JSON action into validated ``Finding`` objects.

Everything here is a pure function of the model output plus the parsed diff,
so the agent loop stays free of finding semantics.
"""

import re
from typing import Iterable, List, Optional

from ..core.diff_parser import ParsedDiff
from ..core.finding_policy import normalize_rule_id
from ..core.models import Finding, Severity
from .loop import collect_evidence


def parse_findings(
    result: dict, parsed: ParsedDiff, role: str,
    validated_skills: Iterable[str] = (),
) -> List[Finding]:
    evidence = collect_evidence(result.get("_observations") or [])
    validated_skills = set(validated_skills)
    findings = []
    for raw in result.get("findings") or []:
        location = resolve_finding_location(raw, parsed)
        if location is None:
            continue
        path, line = location
        try:
            severity = Severity(str(raw.get("severity", "medium")).lower())
        except ValueError:
            severity = Severity.MEDIUM
        refs = [
            evidence[item] for item in raw.get("evidence_ids") or []
            if str(item) in evidence
        ]
        chain = [item for item in (raw.get("call_chain") or []) if isinstance(item, dict)][:20]
        try:
            confidence = float(raw.get("confidence", 0.7))
        except (TypeError, ValueError):
            confidence = 0.7
        original_rule_id = str(raw.get("rule_id", "LLM-OTHER"))[:160]
        rule_id = normalize_model_rule_id(raw)
        claimed_skill = str(raw.get("skill", "")).strip()
        source = (
            "agent-skill:" + claimed_skill
            if claimed_skill in validated_skills else role
        )
        findings.append(Finding(
            rule_id=rule_id,
            severity=severity, title=str(raw.get("title", "Review finding"))[:200],
            explanation=str(raw.get("explanation", ""))[:4000], path=path, line=line,
            evidence=str(raw.get("evidence", ""))[:500],
            fix=str(raw.get("fix", ""))[:4000], test=str(raw.get("test", ""))[:4000],
            confidence=max(0.0, min(1.0, confidence)), evidence_refs=refs,
            call_chain=chain, source=source,
            original_rule_id=(original_rule_id if original_rule_id != rule_id else ""),
        ))
    return findings


def resolve_finding_location(raw: dict, parsed: ParsedDiff) -> Optional[tuple]:
    """Use an exact model location, or uniquely recover it from quoted added code."""
    try:
        path = str(raw.get("path", ""))
        line = int(raw.get("line", 0))
    except (TypeError, ValueError):
        return None
    valid = {(item.path, item.line) for item in parsed.added_lines}
    if (path, line) in valid:
        return path, line
    quoted = str(raw.get("evidence", "")).strip()
    if not path or not quoted or "\n" in quoted:
        return None
    matches = [
        (item.path, item.line)
        for item in parsed.added_lines
        if item.path == path
        and (
            quoted == item.content.strip()
            or quoted in item.content.strip()
        )
    ]
    return matches[0] if len(matches) == 1 else None


def normalize_model_rule_id(raw: dict) -> str:
    """Correct one narrow, auditable CWE mismatch while preserving the raw ID."""
    rule_id = normalize_rule_id(str(raw.get("rule_id", "LLM-OTHER")))
    claim = " ".join((
        str(raw.get("title", "")), str(raw.get("explanation", "")),
        str(raw.get("evidence", "")),
    )).lower()
    if rule_id == "CWE-697" and all((
        any(cue in claim for cue in (
            "canonical", "underscore", "dash", "upload_pack", "upload-pack",
        )),
        any(cue in claim for cue in (
            "bypass", "false negative", "not match", "fails to match",
        )),
        any(cue in claim for cue in (
            "unsafe option", "unsafe_options", "upload_pack", "upload-pack",
        )),
    )):
        return "CWE-184"
    if rule_id != "CWE-252":
        return rule_id
    exception_cues = (
        "exception", "typeerror", "keyerror", "runtimeerror",
        "unboundlocalerror", "raises", "crash",
    )
    return_status_cues = (
        "unchecked return", "return status", "return code",
        "status code", "fails to inspect", "not checked by the caller",
    )
    if (
        any(cue in claim for cue in exception_cues)
        and not any(cue in claim for cue in return_status_cues)
    ):
        return "CWE-248"
    return rule_id


def worker_final_validation_error(result: dict, parsed: ParsedDiff) -> str:
    """Reject positive evidence that would be silently lost during finding parsing."""
    errors = []
    finding_resolutions = {
        str(item.get("evidence_id", ""))
        for item in result.get("evidence_resolutions") or []
        if isinstance(item, dict)
        and str(item.get("status", "")).strip().lower() == "finding"
        and str(item.get("evidence_id", "")).strip()
    }
    valid_citations = set()
    rejected_locations = []
    for raw in result.get("findings") or []:
        if not isinstance(raw, dict):
            continue
        evidence_ids = {
            str(value) for value in raw.get("evidence_ids") or []
            if str(value).strip()
        }
        location = resolve_finding_location(raw, parsed)
        if location is not None:
            valid_citations.update(evidence_ids)
        else:
            rejected_locations.append(
                "%s:%s" % (str(raw.get("path", "")), str(raw.get("line", 0)))
            )

    missing = sorted(finding_resolutions - valid_citations)
    if not missing and not rejected_locations:
        return " ".join(errors)

    allowed = {}
    for item in parsed.added_lines:
        allowed.setdefault(item.path, []).append(item.line)
    allowed_text = "; ".join(
        "%s:%s" % (path, ",".join(str(line) for line in lines[:40]))
        for path, lines in allowed.items()
    )
    errors.append(
        "Every structured Finding must identify an added diff line. The current Finding "
        "would be discarded by validation. Return the same defect anchored to its exact "
        "causal added line. Unmatched positive evidence IDs: %s. "
        "Rejected locations: %s. Valid added locations: %s"
        % (
            ", ".join(missing),
            ", ".join(rejected_locations) or "missing/invalid",
            allowed_text or "none",
        )
    )
    return " ".join(errors)
