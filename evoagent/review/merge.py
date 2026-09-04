"""Pure decisions about a set of findings: normalize, merge, judge, publish.

Every function here is a pure function of its arguments. They hold the rules for
which findings survive a critic, which the Lead may publish and which are
quarantined as unverified suggestions - the part of a review that must stay
inspectable and testable without a model, a store or a network.
"""
import ast
import hashlib
import json
import re
import textwrap
import time
from typing import Iterable, List

from ..agents.loop import collect_evidence
from ..agents.prompts import ROLE_PERMISSIONS
from ..core.diff_parser import ParsedDiff
from ..core.finding_policy import (
    claim_specific_high_risk_evidence_refs, finding_identity,
    is_deterministic_finding, is_validated_agent_skill_finding,
    normalize_rule_id, repository_evidence_refs,
)
from ..core.models import Finding, Severity
from ..session.ledger import ExecutionLedger


def normalize_delegations(
    raw, worker_roles, changed_files, available_skills=None, requested_skills=None,
):
    available_skills = set(available_skills or set())
    requested_skills = [
        name for name in requested_skills or [] if name in available_skills
    ]
    values, seen_ids, covered = [], set(), set()
    for index, item in enumerate(raw or []):
        if not isinstance(item, dict):
            continue
        worker = str(item.get("worker", ""))
        if worker not in worker_roles:
            continue
        assignment_id = str(
            item.get("assignment_id") or "%s-%d" % (worker, index + 1)
        )[:100]
        if not assignment_id or assignment_id in seen_ids:
            continue
        seen_ids.add(assignment_id)
        covered.add(worker)
        values.append({
            "assignment_id": assignment_id, "worker": worker,
            "objective": str(item.get("objective") or "Review the assigned risk domain.")[:2000],
            "files": [str(value)[:500] for value in item.get("files") or changed_files][:100],
            "risk_domains": [str(value)[:100] for value in item.get("risk_domains") or []][:20],
            "required_evidence": [str(value)[:200] for value in item.get("required_evidence") or []][:20],
            "skills": list(dict.fromkeys(requested_skills + [
                str(value) for value in item.get("skills") or []
                if str(value) in available_skills
            ])),
        })
        if len(values) >= 12:
            break
    defaults = {
        "security": "Review security, authorization, input and sensitive-data risks.",
        "correctness-reliability": (
            "Review correctness, failure handling, concurrency, resources and compatibility."
        ),
    }
    for worker in worker_roles:
        if worker in covered or len(values) >= 12:
            continue
        values.append({
            "assignment_id": "%s-default" % worker, "worker": worker,
            "objective": defaults[worker], "files": list(changed_files)[:100],
            "risk_domains": [], "required_evidence": ["changed-line evidence"],
            "skills": list(requested_skills),
        })
    source_suffixes = (
        ".py", ".java", ".kt", ".kts", ".js", ".jsx", ".ts", ".tsx",
        ".go", ".rs", ".rb", ".php", ".cs", ".cpp", ".cc", ".c", ".h",
    )
    production_files = [
        str(path) for path in changed_files
        if str(path).lower().endswith(source_suffixes)
        and not any(
            part in str(path).replace("\\", "/").lower()
            for part in ("/test", "tests/", ".github/", "docs/", "examples/")
        )
    ][:100]
    correctness_files = {
        path
        for item in values
        if item["worker"] == "correctness-reliability"
        for path in item.get("files") or []
    }
    uncovered = [
        path for path in production_files if path not in correctness_files
    ]
    if (
        uncovered
        and "correctness-reliability" in worker_roles
        and len(values) < 12
    ):
        values.append({
            "assignment_id": "correctness-source-coverage",
            "worker": "correctness-reliability",
            "objective": "Review uncovered production source semantics.",
            "files": uncovered[:100],
            "risk_domains": ["correctness"],
            "required_evidence": [
                "Inspect every assigned file with repository evidence.",
            ],
            "skills": list(requested_skills),
        })
    return values


def skill_tool_permissions(worker, skills):
    base = set(ROLE_PERMISSIONS[worker])
    restrictions = [set(skill.allowed_tools) for skill in skills if skill.allowed_tools]
    if not restrictions:
        return base
    return base.intersection(set().union(*restrictions))


def normalize_revision_requests(raw, assignments):
    by_id = {item["assignment_id"]: item for item in assignments}
    values, seen = [], set()
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        assignment_id = str(item.get("assignment_id", ""))
        original = by_id.get(assignment_id)
        if not original or assignment_id in seen:
            continue
        worker = str(item.get("worker") or original["worker"])
        if worker != original["worker"]:
            continue
        guidance = str(item.get("guidance", "")).strip()
        if not guidance:
            continue
        seen.add(assignment_id)
        values.append({
            "assignment_id": assignment_id, "worker": worker,
            "guidance": guidance[:2000],
            "required_evidence": [
                str(value)[:200] for value in item.get("required_evidence") or []
            ][:20],
            "handoff_ids": [
                str(value)[:200] for value in item.get("handoff_ids") or []
                if str(value).strip()
            ][:20],
            "evidence_targets": [
                {
                    "evidence_target_id": str(value.get("evidence_target_id") or "")[:200],
                    "path": str(value.get("path") or "")[:500],
                    "line": int(value.get("line") or 0),
                    "claim": str(value.get("claim") or "")[:2000],
                    "required_proof": str(value.get("required_proof") or "")[:1000],
                    "kind": str(value.get("kind") or "")[:100],
                    "supporting_evidence_ids": [
                        str(item)[:200]
                        for item in value.get("supporting_evidence_ids") or []
                        if str(item).strip()
                    ][:8],
                }
                for value in item.get("evidence_targets") or []
                if isinstance(value, dict) and str(value.get("path") or "").strip()
                and str(value.get("line") or "").isdigit()
            ][:12],
        })
    return values


def candidates_from(rule_findings, worker_results):
    """Scanner findings plus everything the workers reported, deduplicated."""
    findings = list(rule_findings)
    for result in worker_results.values():
        findings.extend(restore_findings(result.get("findings") or []))
    return merge_findings(findings)


def apply_critic(result, candidates):
    evidence = collect_evidence(result.get("_observations") or [])
    by_index = {
        int(item.get("finding_index")): item
        for item in result.get("decisions") or []
        if isinstance(item, dict) and str(item.get("finding_index", "")).isdigit()
    }
    decisions = []
    for index, finding in enumerate(candidates):
        decision = by_index.get(index)
        objections = [
            str(value).strip()
            for value in (decision or {}).get("objections") or []
            if str(value).strip()
        ]
        # ``objections`` is the Critic's list of blocking reasons.  Treating a
        # response as accepted while that list is non-empty made the structured
        # verdict contradict its own explanation and could leak false positives.
        accepted = bool(decision and decision.get("accepted") and not objections)
        verification = {
            key: bool(decision and decision.get(key))
            for key in (
                "introduced_by_diff", "reproducible",
                "evidence_sufficient", "would_comment_on_real_pr",
            )
        }
        publication_ready = accepted and all(verification.values())
        corrected_rule_id = ""
        # Deterministic scanners own their stable rule identity.  A Critic may
        # correct a model-authored CWE, but must not replace a scanner rule
        # with an unverified taxonomy guess; downstream renderers/evaluators
        # can map that canonical rule deterministically.
        if publication_ready and decision and not is_deterministic_finding(finding):
            proposed_rule_id = str(
                decision.get("corrected_rule_id") or ""
            ).strip().upper()
            if re.fullmatch(r"CWE-\d+", proposed_rule_id):
                normalized_rule_id = normalize_rule_id(proposed_rule_id)
                if normalized_rule_id != finding.rule_id:
                    corrected_rule_id = normalized_rule_id
                    if not finding.original_rule_id:
                        finding.original_rule_id = finding.rule_id
                    finding.rule_id = corrected_rule_id
        recommended_adjustment = 0.0
        if decision:
            try:
                recommended_adjustment = float(
                    decision.get("confidence_adjustment", 0)
                )
            except (TypeError, ValueError):
                recommended_adjustment = 0.0
            finding.evidence_refs.extend(
                evidence[str(value)]
                for value in decision.get("supporting_evidence_ids") or []
                if str(value) in evidence
            )
        decisions.append({
            "finding_index": index, "accepted": accepted,
            "publication_ready": publication_ready,
            "recommended_confidence_adjustment": recommended_adjustment,
            "corrected_rule_id": corrected_rule_id,
            **verification,
            "objections": objections if decision else [
                "critic returned no explicit decision"
            ],
        })
    return candidates, decisions


def apply_lead_final(decision, candidates):
    raw_indices = decision.get("accepted_finding_indices") or []
    accepted_indices = {
        int(value) for value in raw_indices if str(value).isdigit()
        and 0 <= int(value) < len(candidates)
    }
    adjustments = {
        int(item.get("finding_index")): item.get("adjustment", 0)
        for item in decision.get("confidence_adjustments") or []
        if isinstance(item, dict) and str(item.get("finding_index", "")).isdigit()
    }
    accepted = []
    for index in sorted(accepted_indices):
        finding = candidates[index]
        try:
            adjustment = float(adjustments.get(index, 0))
        except (TypeError, ValueError):
            adjustment = 0.0
        finding.confidence = max(0.0, min(1.0, finding.confidence + adjustment))
        accepted.append(finding)
    return accepted


def partition_publication(
    rule_findings, candidates, lead_accepted, critic_decisions,
    repository_available, critic_required=True,
    publish_unverified_suggestions=True,
):
    """Protect deterministic findings and quarantine unverified LLM discoveries."""
    lead_identities = {finding_identity(item) for item in lead_accepted}
    critic_by_index = {
        int(item.get("finding_index")): item
        for item in critic_decisions or []
        if isinstance(item, dict) and str(item.get("finding_index", "")).isdigit()
    }
    published = list(rule_findings)
    suggestions = []
    decisions = []
    for index, finding in enumerate(candidates):
        identity = finding_identity(finding)
        lead_selected = identity in lead_identities
        critic = critic_by_index.get(index) or {}
        if is_deterministic_finding(finding):
            finding.disposition = "confirmed"
            decisions.append({
                "finding_index": index, "rule_id": finding.rule_id,
                "source": finding.source, "disposition": "confirmed",
                "reasons": ["protected deterministic scanner baseline"],
            })
            continue

        if is_validated_agent_skill_finding(finding):
            disposition = "confirmed" if lead_selected else "rejected"
            if lead_selected:
                finding.disposition = "confirmed"
                published.append(finding)
            decisions.append({
                "finding_index": index, "rule_id": finding.rule_id,
                "source": finding.source, "disposition": disposition,
                "reasons": [
                    "explicit validated Agent Skill and Lead selection"
                    if lead_selected else "Lead did not select the Agent Skill finding"
                ],
            })
            continue

        reasons = []
        if not lead_selected:
            reasons.append("Lead did not select the candidate")
        if critic_required and not critic.get("publication_ready"):
            reasons.append("Critic did not complete all publication checks")
        repository_refs = repository_evidence_refs(finding)
        claim_refs = claim_specific_high_risk_evidence_refs(finding)
        severity_adjustment = {}
        critic_ready = bool(
            not critic_required or critic.get("publication_ready")
        )
        critic_fully_verified = bool(
            critic_required
            and critic.get("publication_ready")
            and all(critic.get(key) is True for key in (
                "introduced_by_diff", "reproducible",
                "evidence_sufficient", "would_comment_on_real_pr",
            ))
        )
        impact_claim = " ".join((
            finding.title, finding.explanation, finding.evidence,
        )).lower()
        high_impact_cues = (
            "data loss", "data corruption", "service-wide", "global outage",
            "all requests", "irreversible", "deadlock", "authentication bypass",
        )
        high_impact_supported = bool(
            claim_refs and any(cue in impact_claim for cue in high_impact_cues)
        )
        if (
            finding.severity == Severity.HIGH
            and finding.source == "correctness-reliability"
            and lead_selected
            and critic_ready
            and repository_refs
            and not high_impact_supported
        ):
            finding.severity = Severity.MEDIUM
            severity_adjustment = {
                "from": "high", "to": "medium",
                "reason": (
                    "correctness defect is repository-backed, but high impact "
                    "is not supported by concrete outage, data-loss, or corruption evidence"
                ),
            }
        if not repository_available and not claim_refs:
            reasons.append("repository context is unavailable")
        if not repository_refs:
            reasons.append("no repository-backed tool evidence")
        if finding.severity == Severity.LOW:
            reasons.append(
                "low-severity model finding remains advisory"
            )
        # The Worker's confidence is an early estimate.  Once an independent
        # Critic explicitly verifies all four publication obligations against
        # repository facts, do not make that stale estimate override the
        # evidence verdict.  A bare publication_ready flag is intentionally
        # insufficient so integrations cannot bypass the detailed checks.
        confidence_threshold = 0.7 if (
            claim_refs or (critic_fully_verified and repository_refs)
        ) else 0.8
        if finding.confidence + 1e-9 < confidence_threshold:
            reasons.append(
                "model confidence below stable publication threshold %.2f"
                % confidence_threshold
            )
        normalized_fix = re.sub(
            r"[^a-z0-9]+", " ", str(finding.fix or "").lower()
        ).strip()
        explicitly_no_defect = any(
            cue in impact_claim for cue in (
                "no defect is introduced", "does not introduce a defect",
                "no issue is introduced", "does not introduce an issue",
            )
        )
        no_action_fix = normalized_fix in {
            "no fix needed", "no change needed", "none", "n a",
        }
        if explicitly_no_defect or no_action_fix:
            reasons.append(
                "finding explicitly states that no actionable defect exists"
            )
        hypothetical_scope_claim = any(
            cue in impact_claim for cue in (
                "false positive", "incorrectly reject", "overly broad",
            )
        )
        if (
            finding.source == "correctness-reliability"
            and hypothetical_scope_claim
            and not claim_refs
        ):
            reasons.append(
                "hypothetical rejection claim lacks behavioral or configured-value evidence"
            )
        if (
            finding.severity in {Severity.CRITICAL, Severity.HIGH}
            and not claim_refs
        ):
            reasons.append(
                "high-risk claim lacks behavioral or cross-call evidence"
            )

        if not reasons:
            finding.disposition = "confirmed"
            finding.gate = {
                "lead_selected": True,
                "critic_publication_ready": bool(
                    critic_required and critic.get("publication_ready")
                ),
                "repository_evidence_count": len(repository_refs),
                "claim_specific_evidence_count": len(claim_refs),
                "publication_partition_passed": True,
            }
            published.append(finding)
            disposition = "confirmed"
        elif lead_selected and publish_unverified_suggestions:
            finding.disposition = "suggestion"
            finding.gate = {
                "passed": False, "disposition": "suggestion",
                "reasons": list(reasons),
                "repository_evidence_count": len(repository_refs),
                "claim_specific_evidence_count": len(claim_refs),
            }
            suggestions.append(finding)
            disposition = "suggestion"
        elif lead_selected:
            finding.disposition = "rejected"
            reasons.append("stability profile suppresses unverified suggestions")
            disposition = "rejected"
        else:
            disposition = "rejected"
        decisions.append({
            "finding_index": index, "rule_id": finding.rule_id,
            "original_rule_id": finding.original_rule_id,
            "source": finding.source, "disposition": disposition,
            "reasons": reasons,
            "repository_evidence_ids": [
                item.get("evidence_id") for item in repository_refs
            ],
            "severity_adjustment": severity_adjustment,
        })

    return merge_findings(published), merge_findings(suggestions), decisions


def restore_findings(values):
    findings = []
    for value in values or []:
        try:
            severity = Severity(str(value.get("severity", "medium")))
            findings.append(Finding(
                rule_id=str(value.get("rule_id", "REVIEW")), severity=severity,
                title=str(value.get("title", "Review finding")),
                explanation=str(value.get("explanation", "")),
                path=str(value.get("path", "")), line=int(value.get("line", 0)),
                evidence=str(value.get("evidence", "")), fix=str(value.get("fix", "")),
                test=str(value.get("test", "")), confidence=float(value.get("confidence", 0.7)),
                evidence_refs=list(value.get("evidence_refs") or []),
                call_chain=list(value.get("call_chain") or []),
                source=str(value.get("source", "unknown")),
                original_rule_id=str(value.get("original_rule_id", "")),
                disposition=str(value.get("disposition", "candidate")),
            ))
        except (TypeError, ValueError):
            continue
    return findings


def public_decision(result):
    return {
        key: value for key, value in result.items()
        if not str(key).startswith("_")
    }


def merge_findings(findings: Iterable[Finding]) -> List[Finding]:
    merged = {}
    for finding in findings:
        key = (finding.path, finding.line, finding.rule_id)
        evidence_ids = {
            str(item.get("evidence_id"))
            for item in finding.evidence_refs if isinstance(item, dict)
            and str(item.get("evidence_id", ""))
        }
        semantic_ids = {
            value for value in evidence_ids
            if value.startswith("semantic_probe:")
        }
        if evidence_ids and not is_deterministic_finding(finding):
            title_tokens = set(re.findall(
                r"[a-z0-9_]+", str(finding.title).lower()
            ))
            for existing_key, existing in merged.items():
                existing_ids = {
                    str(item.get("evidence_id"))
                    for item in existing.evidence_refs if isinstance(item, dict)
                    and str(item.get("evidence_id", ""))
                }
                existing_tokens = set(re.findall(
                    r"[a-z0-9_]+", str(existing.title).lower()
                ))
                title_overlap = (
                    len(title_tokens.intersection(existing_tokens))
                    / max(1, min(len(title_tokens), len(existing_tokens)))
                )
                same_probe_claim = bool(
                    semantic_ids.intersection(existing_ids)
                    and (
                        existing.line == finding.line
                        or (
                            existing.rule_id == finding.rule_id
                            and abs(existing.line - finding.line) <= 5
                        )
                    )
                )
                if (
                    existing.path == finding.path
                    and not is_deterministic_finding(existing)
                    and (
                        same_probe_claim
                        or (
                            existing.line == finding.line and
                            evidence_ids.intersection(existing_ids)
                            and title_overlap >= 0.6
                        )
                    )
                ):
                    key = existing_key
                    break
        current = merged.get(key)
        priority = (
            2 if is_deterministic_finding(finding)
            else 1 if is_validated_agent_skill_finding(finding) else 0
        )
        current_priority = (
            2 if current is not None and is_deterministic_finding(current)
            else 1 if current is not None and is_validated_agent_skill_finding(current)
            else 0
        )
        claim_supported = bool(
            claim_specific_high_risk_evidence_refs(finding)
        )
        current_claim_supported = bool(
            current is not None
            and claim_specific_high_risk_evidence_refs(current)
        )
        if (
            current is None or priority > current_priority
            or (
                priority == current_priority
                and claim_supported
                and not current_claim_supported
            )
            or (
                priority == current_priority
                and claim_supported == current_claim_supported
                and finding.confidence > current.confidence
            )
        ):
            if current is not None:
                known = {
                    str(item.get("evidence_id")) for item in finding.evidence_refs
                    if isinstance(item, dict)
                }
                finding.evidence_refs.extend(
                    item for item in current.evidence_refs
                    if isinstance(item, dict)
                    and str(item.get("evidence_id")) not in known
                )
            merged[key] = finding
        elif current is not None:
            known = {
                str(item.get("evidence_id")) for item in current.evidence_refs
                if isinstance(item, dict)
            }
            current.evidence_refs.extend(
                item for item in finding.evidence_refs
                if isinstance(item, dict)
                and str(item.get("evidence_id")) not in known
            )
    order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}
    return sorted(merged.values(), key=lambda item: (order[item.severity], item.path, item.line))


def scanner_name(name: str) -> str:
    value = str(name)
    return value[:-5] + "scanner" if value.endswith("-agent") else value


def attach_diff_ast_evidence(
    findings: List[Finding], parsed: ParsedDiff, ledger: ExecutionLedger,
) -> int:
    lines = {(item.path, item.line): item.content for item in parsed.added_lines}
    scans = 0
    for finding in findings:
        if finding.severity not in {Severity.CRITICAL, Severity.HIGH}:
            continue
        source = lines.get((finding.path, finding.line), "")
        if not finding.path.endswith(".py") or not source.strip():
            continue
        started = time.monotonic()
        try:
            tree = ast.parse(textwrap.dedent(source))
            structures = [
                type(node).__name__ for node in ast.walk(tree)
                if isinstance(node, (ast.Call, ast.Assign, ast.AnnAssign, ast.keyword))
            ]
            supported = bool(structures)
            payload = {
                "path": finding.path, "line": finding.line,
                "valid_python_ast": True, "structures": structures,
                "rule_id": finding.rule_id,
            }
        except SyntaxError as exc:
            supported = False
            payload = {
                "path": finding.path, "line": finding.line,
                "valid_python_ast": False, "error": str(exc),
            }
        ledger.record_tool(
            "agentic-scanner", "diff-ast-analyze",
            {"path": finding.path, "line": finding.line}, supported,
            int((time.monotonic() - started) * 1000), payload,
            "" if supported else payload.get("error", "no relevant AST structure"),
        )
        scans += 1
        if supported:
            rendered = json.dumps(payload, sort_keys=True)
            finding.evidence_refs.append({
                "evidence_id": "diff-ast:%s" % hashlib.sha256(
                    rendered.encode("utf-8")
                ).hexdigest()[:16],
                "tool": "diff-ast-analyze", **payload,
            })
    return scans
