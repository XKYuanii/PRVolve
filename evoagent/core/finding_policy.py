"""Canonical rule identities and conservative publication policy helpers."""
import json
import re
from typing import List

from ..core.models import Finding


CANONICAL_RULE_IDS = frozenset({
    "SEC-EVAL",
    "SEC-SUBPROCESS-SHELL",
    "SEC-HARDCODED-SECRET",
    "SEC-SQL-CONCAT",
    "REL-EMPTY-EXCEPT",
    "REL-DEBUG-PRINT",
    "SEC-PATH-TRAVERSAL",
    "SEC-YAML-LOAD",
    "SEC-WEAK-HASH",
    "SEC-INSECURE-TEMPFILE",
    "SEC-WEAK-RANDOM",
    "REL-UNBOUNDED-RETRY",
    "SEC-ASSERT-AUTH",
    "SEC-INSECURE-COOKIE",
    "SEC-PICKLE-LOAD",
    "REL-FLOAT-MONEY",
    "REL-NAIVE-DATETIME",
    "REL-BLOCKING-ASYNC",
    "REL-NONATOMIC-WRITE",
    "SEC-OPEN-REDIRECT",
    "SEC-LOG-FORGING",
    "SEC-JINJA-UNSANDBOXED",
    "SEC-JWT-SIGNATURE-DISABLED",
    "SEC-GHA-EXPRESSION-IN-SHELL",
})

RULE_ID_ALIASES = {
    "INJECTION-PICKLE-UNTRUSTED": "SEC-PICKLE-LOAD",
    "UNSAFE-PICKLE-DESERIALIZATION": "SEC-PICKLE-LOAD",
    "UNSAFE-DESERIALIZATION-PICKLE": "SEC-PICKLE-LOAD",
    "DYNAMIC-CODE-EXECUTION": "SEC-EVAL",
    "COMMAND-INJECTION-SHELL-TRUE": "SEC-SUBPROCESS-SHELL",
    "HARDCODED-SECRET": "SEC-HARDCODED-SECRET",
    "SQL-INJECTION-CONCAT": "SEC-SQL-CONCAT",
}

GENERIC_MODEL_RULE = re.compile(
    r"^(?:CORRECTNESS|RELIABILITY|SECURITY|SEC|REL|CR|BUG|ISSUE)(?:-|_)?\d+$"
)
VALID_RULE_ID = re.compile(r"^[A-Z][A-Z0-9_-]{1,79}$")

DETERMINISTIC_SOURCE_PREFIXES = (
    "local-rule-scanner",
    "declarative-scanner:",
)

REPOSITORY_EVIDENCE_TOOLS = frozenset({
    "search_repository",
    "read_file",
    "symbol",
    "locate_tests",
    "read_project_controls",
    "ast_analyze",
    "git_context",
    "run_scanners",
    "run_repository_checks",
    "semantic_probe",
    "semgrep",
    "bandit",
    "eslint",
    "typecheck",
    "test",
})

BEHAVIORAL_EVIDENCE_TOOLS = frozenset({
    "semantic_probe", "run_repository_checks", "test", "run_scanners",
    "semgrep", "bandit", "eslint", "typecheck",
})


def normalize_rule_id(value: str) -> str:
    """Normalize known aliases without destroying valid dynamic Skill rule ids."""
    rendered = re.sub(r"[^A-Za-z0-9]+", "-", str(value or "").strip()).strip("-")
    rendered = rendered.upper()[:80]
    if not rendered:
        return "LLM-OTHER"
    rendered = RULE_ID_ALIASES.get(rendered, rendered)
    if rendered in CANONICAL_RULE_IDS:
        return rendered
    if GENERIC_MODEL_RULE.fullmatch(rendered) or not VALID_RULE_ID.fullmatch(rendered):
        return "LLM-OTHER"
    return rendered


def is_deterministic_finding(finding: Finding) -> bool:
    source = str(finding.source or "")
    return source.startswith(DETERMINISTIC_SOURCE_PREFIXES)


def is_validated_agent_skill_finding(finding: Finding) -> bool:
    return str(finding.source or "").startswith("agent-skill:")


def repository_evidence_refs(finding: Finding) -> List[dict]:
    return [
        item for item in finding.evidence_refs
        if isinstance(item, dict)
        and item.get("evidence_id")
        and str(item.get("tool")) in REPOSITORY_EVIDENCE_TOOLS
        and _repository_evidence_has_facts(item)
    ]


def claim_specific_high_risk_evidence_refs(finding: Finding) -> List[dict]:
    """Return evidence strong enough to publish a model's high-risk claim.

    Merely proving that the changed line exists, or that its file parses, does
    not prove the claimed failure mode. Accept either bounded behavioral tool
    output or repository evidence that corroborates a different node in the
    model's explicit call chain.
    """
    refs = repository_evidence_refs(finding)
    supported = [
        item for item in refs
        if str(item.get("tool")) in BEHAVIORAL_EVIDENCE_TOOLS
        and (
            str(item.get("tool")) != "semantic_probe"
            or _semantic_probe_supports_finding(item, finding)
        )
    ]
    origin = (_normalized_evidence_path(finding.path), int(finding.line))
    chain = []
    for raw in finding.call_chain or []:
        if not isinstance(raw, dict) or not raw.get("path"):
            continue
        try:
            location = (
                _normalized_evidence_path(str(raw["path"])), int(raw.get("line", 0)),
            )
        except (TypeError, ValueError):
            continue
        if location[1] > 0 and location != origin:
            chain.append(location)
    if not chain:
        return supported
    for item in refs:
        if item in supported:
            continue
        locations = _evidence_locations(item)
        if any(
            path == chain_path and abs(line - chain_line) <= 3
            for path, line in locations
            for chain_path, chain_line in chain
        ):
            supported.append(item)
    return supported


def _semantic_probe_supports_finding(item: dict, finding: Finding) -> bool:
    payload = item.get("output")
    if payload is None:
        payload = item.get("output_preview")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
    if not isinstance(payload, dict):
        return False
    kind = str(payload.get("kind", "")).strip().lower()
    claim = " ".join((
        str(finding.rule_id), str(finding.title), str(finding.explanation),
        str(finding.evidence),
    )).lower()
    if kind == "path-containment" and any(
        cue in claim for cue in ("symlink", "evalsymlinks", "isnotexist", "toctou")
    ):
        # This probe only proves lexical parent-segment escape after Join. It
        # does not create symlinks or exercise filesystem race/error branches.
        return False
    if kind == "git-option-normalization" and any(
        cue in claim for cue in (
            "false positive", "incorrectly reject", "overly broad", "config-file",
        )
    ):
        # This probe proves underscore-to-dash bypass behavior only. It does not
        # prove that any configured unsafe option rejects a legitimate longer
        # option by prefix.
        return False
    cues = {
        "path-containment": ("cwe-22", "path traversal", "containment", "filepath"),
        "security-control-default": (
            "cwe-287", "cwe-347", "signature", "verify_signature", "authentication",
        ),
        "github-actions-expression-shell": (
            "cwe-78", "cwe-94", "command injection", "shell", "github actions",
            "expression interpolation",
        ),
        "git-option-normalization": (
            "cwe-78", "cwe-184", "unsafe option", "upload_pack", "upload-pack",
            "underscore", "canonical",
        ),
        "url-normalization-redaction": (
            "url", "redact", "credential", "cwe-200", "cwe-522", "cwe-532",
        ),
        "tri-state-boolean": ("tri-state", "boolean", "none", "default"),
        "serialization-exclusion-update": (
            "serializ", "model_dump", "exclude", "update",
        ),
        "equality-negation-contract": ("__eq__", "__ne__", "equality", "negation"),
        "decorator-order": ("decorator", "classmethod"),
        "self-cycle-collection": ("cycle", "garbage collect", "gc"),
        "alias-configuration-direction": (
            "validation_alias", "validate_by_alias", "alias",
        ),
        "dict-mutation-during-iteration": (
            "dictionary changed size", "dict", "pop(", "runtimeerror",
        ),
        "decimal-special-exponent": (
            "decimal", "nan", "infinity", "exponent", "typeerror",
        ),
        "unhashable-exception-membership": (
            "unhashable", "exception", "set", "typeerror",
        ),
        "scandir-missing-directory": (
            "scandir", "filenotfounderror", "missing directory",
            "directory disappeared", "directory removed",
        ),
        "empty-netrc-credentials": (
            "netrc", "empty credentials", "blank credentials", "tuple",
        ),
        "exception-cleanup-state": (
            "cleanup", "state", "redirect", "render hook", "refresh",
            "exception", "cwe-404", "cwe-459", "cwe-664", "cwe-703",
            "cwe-772",
        ),
    }
    return kind in cues and any(cue in claim for cue in cues[kind])


def _normalized_evidence_path(path: str) -> str:
    value = str(path or "").replace("\\", "/").strip()
    return value[2:] if value.startswith(("a/", "b/")) else value


def _evidence_locations(item: dict) -> List[tuple]:
    payload = item.get("output")
    if payload is None:
        payload = item.get("output_preview")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
    values = payload if isinstance(payload, list) else [payload]
    locations = []
    for value in values:
        if not isinstance(value, dict) or not value.get("path"):
            continue
        path = _normalized_evidence_path(str(value["path"]))
        if value.get("line") is not None:
            try:
                locations.append((path, int(value["line"])))
            except (TypeError, ValueError):
                pass
            continue
        try:
            start = int(value.get("start_line", 0))
            end = int(value.get("end_line", start))
        except (TypeError, ValueError):
            continue
        if start > 0 and end >= start:
            locations.extend((path, line) for line in range(start, end + 1))
    return locations


def _repository_evidence_has_facts(item: dict) -> bool:
    """Reject empty/unavailable tool calls masquerading as supporting evidence."""
    payload = item.get("output")
    if payload is None:
        payload = item.get("output_preview")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = payload.strip()
    tool = str(item.get("tool"))
    if payload in (None, "", [], {}):
        return False
    if tool in {"search_repository", "locate_tests", "read_project_controls"}:
        return isinstance(payload, list) and bool(payload)
    if tool == "read_file":
        return isinstance(payload, dict) and bool(str(payload.get("content", "")).strip())
    if tool == "symbol":
        return isinstance(payload, dict) and any(
            payload.get(key) for key in ("definitions", "callers", "callees")
        )
    if tool == "ast_analyze":
        return isinstance(payload, dict) and payload.get("valid") is True
    if tool == "semantic_probe":
        return (
            isinstance(payload, dict)
            and bool(payload.get("kind"))
            and payload.get("arbitrary_code_executed") is False
        )
    if tool == "git_context":
        return (
            isinstance(payload, dict)
            and payload.get("available") is True
            and any(str(value.get("output", "")).strip() for value in payload.get("results") or [])
        )
    if tool in {"run_scanners", "semgrep", "bandit", "eslint", "typecheck"}:
        return (
            isinstance(payload, dict)
            and payload.get("available") is not False
            and any(
                run.get("available") is True and bool(str(run.get("output", "")).strip())
                for run in payload.get("runs") or []
                if isinstance(run, dict)
            )
        )
    if tool in {"run_repository_checks", "test"}:
        return (
            isinstance(payload, dict)
            and payload.get("available") is True
            and ("passed" in payload or "returncode" in payload)
        )
    return True


def finding_identity(finding: Finding) -> tuple:
    return finding.path, int(finding.line), finding.rule_id
