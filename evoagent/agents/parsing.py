"""Turn a model's raw JSON action into validated ``Finding`` objects.

Everything here is a pure function of the model output plus the parsed diff,
so the agent loop stays free of finding semantics.
"""

import re
from typing import Dict, Iterable, List, Optional

from ..core.diff_parser import ParsedDiff
from ..core.finding_policy import normalize_rule_id
from ..core.models import Finding, Severity
from .loop import (
    REFUTATION_PROOF_KINDS, collect_evidence, refutation_supported,
)


HYPOTHESIS_STATUSES = {"finding", "refuted", "unresolved", "handoff"}
REQUIREMENT_STATUSES = HYPOTHESIS_STATUSES | {"satisfied"}
WORKER_ROLES = {"security", "correctness-reliability"}
SCOPE_ONLY_REFUTATION_CUES = (
    "outside my scope", "out of scope", "not my responsibility",
    "belongs to the other worker", "belongs to correctness",
    "belongs to security", "not a security issue", "not security-relevant",
    "not a correctness issue", "no callers found", "could not find a caller",
    "no counterexample found", "could not find a counterexample",
)


def parse_findings(
    result: dict, parsed: ParsedDiff, role: str,
    validated_skills: Iterable[str] = (),
    recalled_lesson_ids: Iterable[str] = (),
) -> List[Finding]:
    evidence = collect_evidence(result.get("_observations") or [])
    validated_skills = set(validated_skills)
    recalled_lesson_ids = {
        str(value) for value in recalled_lesson_ids if str(value)
    }
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
        # A model may declare which recalled lessons influenced this claim, but
        # it cannot invent provenance or use a lesson as factual evidence.
        raw_used_lesson_ids = raw.get("used_lesson_ids")
        if not isinstance(raw_used_lesson_ids, list):
            raw_used_lesson_ids = []
        used_lesson_ids = list(dict.fromkeys(
            str(value) for value in raw_used_lesson_ids
            if str(value) in recalled_lesson_ids
        ))[:20]
        findings.append(Finding(
            rule_id=rule_id,
            severity=severity, title=str(raw.get("title", "Review finding"))[:200],
            explanation=str(raw.get("explanation", ""))[:4000], path=path, line=line,
            evidence=str(raw.get("evidence", ""))[:500],
            fix=str(raw.get("fix", ""))[:4000], test=str(raw.get("test", ""))[:4000],
            confidence=max(0.0, min(1.0, confidence)), evidence_refs=refs,
            call_chain=chain, source=source,
            original_rule_id=(original_rule_id if original_rule_id != rule_id else ""),
            used_lesson_ids=used_lesson_ids,
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
    path = str(raw.get("path") or "").replace("\\", "/").lower()
    if (
        path.endswith(".py")
        and rule_id in {"CWE-787", "CWE-476"}
        and "indexerror" in claim
        and any(cue in claim for cue in (
            "index", "out-of-bounds", "out of bounds", "list", "sequence",
        ))
    ):
        # Python list indexing raises a managed IndexError; it is not a native
        # out-of-bounds write or null dereference. Preserve the raw label in
        # original_rule_id while using the relevant bounds taxonomy.
        return "CWE-129"
    if (
        path.endswith(".py")
        and rule_id == "CWE-476"
        and "keyerror" in claim
        and any(cue in claim for cue in (
            "mapping", "dictionary", "dict", "missing key", "direct indexing",
            "subscript", ".get(", "['", '["',
        ))
    ):
        # A Python mapping subscript raises KeyError; it is not a null-pointer
        # dereference. Keep the model's original label for the audit trail.
        return "CWE-248"
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
    if (
        not any(cue in claim for cue in return_status_cues)
        and any(cue in claim for cue in (
            "mask all", "masks all", "all data", "all values", "wrong result",
        ))
        and any(cue in claim for cue in ("mask", "nan", "filter"))
    ):
        # No return value was ignored: the defect is the computed data value.
        return "CWE-682"
    return rule_id


def assignment_requirements(assignment: Optional[dict]) -> List[dict]:
    """Give every Lead requirement a stable ID for one Worker run."""
    values = []
    seen = set()
    for raw in (assignment or {}).get("required_evidence") or []:
        question = str(raw).strip()
        if not question or question in seen:
            continue
        seen.add(question)
        values.append({
            "requirement_id": "req-%d" % (len(values) + 1),
            "question": question[:500],
        })
    return values[:20]


def normalize_hypotheses(raw) -> List[dict]:
    values = []
    for index, item in enumerate(raw or []):
        if not isinstance(item, dict):
            continue
        values.append({
            "hypothesis_id": str(
                item.get("hypothesis_id") or "hyp-%d" % (index + 1)
            )[:100],
            "claim": str(item.get("claim") or item.get("hypothesis") or "")[:2000],
            "location": str(item.get("location") or "")[:500],
            "domain": str(item.get("domain") or "")[:100],
            "risk_level": str(item.get("risk_level") or "normal")[:20].lower(),
            "status": str(item.get("status") or "").strip().lower()[:40],
            "explanation": str(item.get("explanation") or "")[:2000],
            "proof_kind": str(item.get("proof_kind") or "").strip().lower()[:80],
            "supporting_evidence_ids": [
                str(value)[:200]
                for value in item.get("supporting_evidence_ids") or []
                if str(value).strip()
            ][:20],
            "required_proof": str(item.get("required_proof") or "")[:1000],
            "target_worker": str(item.get("target_worker") or "")[:100],
            "origin": str(item.get("origin") or "worker")[:80],
        })
    return values[:30]


def normalize_requirement_resolutions(raw) -> List[dict]:
    values = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        values.append({
            "requirement_id": str(item.get("requirement_id") or "")[:100],
            "status": str(item.get("status") or "").strip().lower()[:40],
            "explanation": str(item.get("explanation") or "")[:2000],
            "proof_kind": str(item.get("proof_kind") or "").strip().lower()[:80],
            "supporting_evidence_ids": [
                str(value)[:200]
                for value in item.get("supporting_evidence_ids") or []
                if str(value).strip()
            ][:20],
            "required_proof": str(item.get("required_proof") or "")[:1000],
            "target_worker": str(item.get("target_worker") or "")[:100],
            "origin": str(item.get("origin") or "worker")[:80],
        })
    return values[:20]


def downgrade_unsupported_conclusions(result: dict) -> List[dict]:
    """Turn unsupported certainty into explicit uncertainty without losing the claim."""
    evidence = collect_evidence(result.get("_observations") or [])
    changes = []
    collections = (
        ("hypotheses", HYPOTHESIS_STATUSES),
        ("requirement_resolutions", REQUIREMENT_STATUSES),
    )
    for collection, allowed in collections:
        for index, item in enumerate(result.get(collection) or []):
            if not isinstance(item, dict):
                continue
            status = str(item.get("status") or "").strip().lower()
            label = "%s[%d]" % (collection, index)
            error = ""
            if status in {"refuted", "satisfied"}:
                error = _resolution_error(item, label, allowed, evidence)
            elif status == "finding" and not (result.get("findings") or []):
                error = "%s has status finding but no structured Finding." % label
            if not error:
                continue
            item["status"] = "unresolved"
            if collection == "hypotheses":
                item["risk_level"] = "high"
            item["origin"] = "protocol-downgrade"
            item["required_proof"] = error[:1000]
            explanation = str(item.get("explanation") or "").strip()
            item["explanation"] = (
                (explanation + " ").strip()
                + " Protocol downgraded this conclusion because its certainty was unsupported."
            )[:2000]
            changes.append({
                "collection": collection, "index": index,
                "from_status": status, "to_status": "unresolved",
                "reason": error[:1000],
            })
    return changes


def worker_handoffs(
    hypotheses: List[dict], requirement_resolutions: List[dict], assignment: dict,
) -> List[dict]:
    """Project cross-domain observations into records the existing Lead can route."""
    assignment_id = str(assignment.get("assignment_id") or "")
    source_worker = str(assignment.get("worker") or "")
    requirements = {
        item["requirement_id"]: item["question"]
        for item in assignment_requirements(assignment)
    }
    values = []
    for item in hypotheses:
        if item.get("status") != "handoff":
            continue
        item_id = str(item.get("hypothesis_id") or "hypothesis")
        values.append({
            "handoff_id": "%s:%s" % (assignment_id, item_id),
            "source_assignment_id": assignment_id,
            "source_worker": source_worker,
            "target_worker": item.get("target_worker", ""),
            "claim": item.get("claim", ""),
            "location": item.get("location", ""),
            "explanation": item.get("explanation", ""),
            "required_proof": item.get("required_proof", ""),
            "supporting_evidence_ids": item.get("supporting_evidence_ids", []),
            "origin": item.get("origin", "worker"),
        })
    for item in requirement_resolutions:
        if item.get("status") != "handoff":
            continue
        item_id = str(item.get("requirement_id") or "requirement")
        values.append({
            "handoff_id": "%s:%s" % (assignment_id, item_id),
            "source_assignment_id": assignment_id,
            "source_worker": source_worker,
            "target_worker": item.get("target_worker", ""),
            "claim": requirements.get(item_id, item_id),
            "location": "",
            "explanation": item.get("explanation", ""),
            "required_proof": item.get("required_proof", ""),
            "supporting_evidence_ids": item.get("supporting_evidence_ids", []),
            "origin": item.get("origin", "worker"),
        })
    return values


def _resolution_error(
    item: Dict[str, object], label: str, allowed_statuses, evidence,
) -> str:
    status = str(item.get("status") or "").strip().lower()
    explanation = str(item.get("explanation") or "").strip()
    supporting = {
        str(value) for value in item.get("supporting_evidence_ids") or []
        if str(value).strip()
    }
    cited = supporting.intersection(evidence)
    if status not in allowed_statuses:
        return "%s has invalid status %r." % (label, status)
    if not explanation:
        return "%s requires an explanation." % label
    lowered = explanation.lower()
    if status == "refuted":
        proof_kind = str(item.get("proof_kind") or "").strip().lower()
        if proof_kind not in REFUTATION_PROOF_KINDS:
            return (
                "%s may be refuted only with proof_kind %s."
                % (label, ", ".join(sorted(REFUTATION_PROOF_KINDS)))
            )
        if any(cue in lowered for cue in SCOPE_ONLY_REFUTATION_CUES):
            return (
                "%s cannot be refuted because it belongs to another domain or because "
                "no counterexample/caller was found; use handoff or unresolved." % label
            )
        if not cited:
            return "%s refutation must cite successful repository evidence." % label
        if not refutation_supported(proof_kind, supporting, evidence):
            return (
                "%s cited evidence does not establish the claimed %s proof."
                % (label, proof_kind)
            )
    elif status == "satisfied":
        if not cited:
            return "%s satisfaction must cite successful evidence." % label
    elif status == "unresolved":
        if not str(item.get("required_proof") or "").strip():
            return "%s unresolved status requires required_proof." % label
    elif status == "handoff":
        target = str(item.get("target_worker") or "")
        if target not in WORKER_ROLES:
            return "%s handoff requires a valid target_worker." % label
        if not str(item.get("required_proof") or "").strip():
            return "%s handoff requires required_proof for the receiving Worker." % label
    return ""


def worker_final_validation_error(
    result: dict, parsed: ParsedDiff, demand_hypotheses: bool = False,
    assignment: Optional[dict] = None, repository_available: bool = False,
) -> str:
    """Enforce a lossless Worker conclusion protocol and valid Finding anchors."""
    errors = []
    evidence = collect_evidence(result.get("_observations") or [])
    hypotheses = normalize_hypotheses(result.get("hypotheses"))
    requirement_resolutions = normalize_requirement_resolutions(
        result.get("requirement_resolutions")
    )
    if demand_hypotheses and not (result.get("findings") or []):
        if not hypotheses:
            errors.append(
                "Returning no findings requires hypotheses. List every risk this change "
                "raised - especially conditions, checks or early returns it removed - with "
                "a status of finding, refuted, unresolved or handoff. Use refuted only with an "
                "invariant, assertion, exhaustive write-path enumeration or test that makes "
                "the risk impossible; not having found a counterexample is unresolved."
            )
    for item in hypotheses:
        label = "Hypothesis %s" % (item.get("hypothesis_id") or "without an ID")
        if not str(item.get("claim") or "").strip():
            errors.append("%s requires a claim." % label)
            continue
        error = _resolution_error(
            item, label, HYPOTHESIS_STATUSES, evidence,
        )
        if error:
            errors.append(error)
        if (
            item.get("status") == "handoff"
            and assignment
            and item.get("target_worker") == assignment.get("worker")
        ):
            errors.append("%s handoff must target the other Worker." % label)
        if item.get("status") == "finding" and not (result.get("findings") or []):
            errors.append(
                "%s has status finding but no structured Finding was returned." % label
            )

    requirements = assignment_requirements(assignment)
    by_requirement = {
        str(item.get("requirement_id") or ""): item
        for item in requirement_resolutions
        if item.get("requirement_id")
    }
    missing_requirements = [
        item["requirement_id"] for item in requirements
        if item["requirement_id"] not in by_requirement
    ]
    if missing_requirements:
        errors.append(
            "Resolve every Lead assignment requirement exactly once. Missing: %s."
            % ", ".join(missing_requirements)
        )
    requirement_ids = {item["requirement_id"] for item in requirements}
    unknown_requirements = sorted(
        item_id for item_id in by_requirement if item_id not in requirement_ids
    )
    if unknown_requirements:
        errors.append(
            "Unknown requirement_resolutions IDs: %s."
            % ", ".join(unknown_requirements)
        )
    duplicates = sorted({
        item_id for item_id in by_requirement
        if sum(
            str(value.get("requirement_id") or "") == item_id
            for value in requirement_resolutions
        ) > 1
    })
    if duplicates:
        errors.append(
            "Resolve each Lead requirement exactly once. Duplicates: %s."
            % ", ".join(duplicates)
        )
    for requirement in requirements:
        item = by_requirement.get(requirement["requirement_id"])
        if item is None:
            continue
        error = _resolution_error(
            item, "Requirement %s" % requirement["requirement_id"],
            REQUIREMENT_STATUSES, evidence,
        )
        if error:
            errors.append(error)
        if (
            item.get("status") == "handoff"
            and assignment
            and item.get("target_worker") == assignment.get("worker")
        ):
            errors.append(
                "Requirement %s handoff must target the other Worker."
                % requirement["requirement_id"]
            )
        if item.get("status") == "finding" and not (result.get("findings") or []):
            errors.append(
                "Requirement %s has status finding but no structured Finding was returned."
                % requirement["requirement_id"]
            )
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
