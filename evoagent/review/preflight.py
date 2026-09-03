"""Prefetch risk-ranked repository context before a worker's first model call.

Workers that start blind burn their first turns on orientation. This runs the
read-only tools up front and hands the results in as observations, so the first
model call already carries the source, tests and call sites the assignment is
about.
"""
import re


def repository_preflight(
    assignment, parsed, tools, repository_available=True,
):
    """Prefetch risk-ranked source context before a worker's first model call."""
    files = set(assignment.get("files") or parsed.files)
    added = [item for item in parsed.added_lines if item.path in files]
    risk_cues = {
        "__eq__", "__ne__", "classmethod", "staticmethod", "except",
        "hasattr", "len(", "model_dump", "none", "pop(", "replace(",
        "secret", "token", "validation_alias", "warn(", "warning",
        "weakref", "_gc_cycle", "redirect", "normalize", "decode", "__getattr__",
        "environment(", "sandboxedenvironment", "_inline_env",
    }

    def priority(item):
        path = item.path.replace("\\", "/").lower()
        content = item.content.lower()
        score = 20 if path.endswith(".py") else 0
        if content.lstrip().startswith(("import ", "from ")):
            score -= 20
        if not any(part in path for part in ("/test", "tests/", ".github/", "docs/")):
            score += 15
        score += 8 * sum(cue in content for cue in risk_cues)
        return (-score, path, item.line)

    added.sort(key=priority)
    ignored = {
        "append", "format", "get", "items", "join", "strip", "replace",
        "self", "true", "false", "none", "return", "import", "from",
        "else", "with", "line", "value", "result", "object", "string",
        "info", "logger", "warning", "warn",
    }
    observations = []
    selected_regions = []
    seen_regions = set()
    for item in added:
        region = (item.path, max(0, (item.line - 1) // 40))
        if region in seen_regions:
            continue
        seen_regions.add(region)
        selected_regions.append(item)
        if len(selected_regions) >= 3:
            break
    selected = selected_regions[0] if selected_regions else None
    if repository_available and selected_regions and "read_file" in tools.names():
        for region_item in selected_regions:
            try:
                value = tools.invoke("read_file", {
                    "path": region_item.path,
                    "start_line": max(1, region_item.line - 25),
                    "end_line": region_item.line + 25,
                })
                observations.append({
                    "step": 0, "tool": "read_file", "ok": True, "result": value,
                    "reason": "evidence-first risk-ranked source context",
                })
            except Exception as exc:
                observations.append({
                    "step": 0, "tool": "read_file", "ok": False,
                    "error": str(exc)[:1000],
                })
    queries = []
    cross_file_hit = None
    text = ""
    if selected:
        nearby = [
            item.content for item in added
            if item.path == selected.path and abs(item.line - selected.line) <= 12
        ]
        text = "\n".join(nearby or [selected.content])
        attributes = re.findall(r"\.\s*([A-Za-z_][A-Za-z0-9_]*)", text)
        identifiers = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b", text)
        ranked = attributes + sorted(
            identifiers,
            key=lambda value: (
                not ("_" in value or value.isupper()), -len(value), value,
            ),
        )
        for query in ranked:
            if query.lower() in ignored or query in queries:
                continue
            queries.append(query)
            if len(queries) >= 2:
                break
    if repository_available and "search_repository" in tools.names():
        for query in queries:
            try:
                value = tools.invoke(
                    "search_repository", {"query": query, "limit": 10}
                )
                observations.append({
                    "step": 0, "tool": "search_repository", "ok": True,
                    "result": value, "reason": "evidence-first semantic contract probe",
                })
                payload = value.get("output") if isinstance(value, dict) else None
                if cross_file_hit is None and isinstance(payload, list):
                    candidates = [
                        item for item in payload
                        if isinstance(item, dict)
                        and item.get("path") != selected.path
                        and not str(item.get("content", "")).lstrip().startswith(
                            ("import ", "from ")
                        )
                    ]
                    if candidates:
                        cross_file_hit = candidates[0]
            except Exception as exc:
                observations.append({
                    "step": 0, "tool": "search_repository", "ok": False,
                    "error": str(exc)[:1000],
                })
    if (
        repository_available and cross_file_hit
        and "read_file" in tools.names()
    ):
        try:
            cross_line = max(1, int(cross_file_hit.get("line", 1)))
            value = tools.invoke("read_file", {
                "path": str(cross_file_hit["path"]),
                "start_line": max(1, cross_line - 20),
                "end_line": cross_line + 30,
            })
            observations.append({
                "step": 0, "tool": "read_file", "ok": True,
                "result": value,
                "reason": "evidence-first cross-file use-site context",
            })
            payload = value.get("output") if isinstance(value, dict) else None
            content = str(payload.get("content", "")) if isinstance(payload, dict) else ""
            declarations = re.findall(
                r"^\s*(?:class|def|async\s+def)\s+([A-Za-z_][A-Za-z0-9_]*)",
                content, flags=re.MULTILINE,
            )
            if declarations and "search_repository" in tools.names():
                value = tools.invoke("search_repository", {
                    "query": declarations[0], "limit": 10,
                })
                observations.append({
                    "step": 0, "tool": "search_repository", "ok": True,
                    "result": value,
                    "reason": "evidence-first one-hop caller search",
                })
        except Exception as exc:
            observations.append({
                "step": 0, "tool": "read_file", "ok": False,
                "error": str(exc)[:1000],
            })
    probe_kinds = []
    # Probe all bounded added text rather than only the first selected
    # region; large PRs commonly place the relevant contract in a later
    # hunk of the same file.
    semantic_text = "\n".join(item.content for item in added[:500])
    lowered = semantic_text.lower()
    if any(token in lowered for token in ("filepath.join", "os.path.join")) and any(
        token in lowered for token in ("repodir", "repository", "base", "path")
    ):
        probe_kinds.append("path-containment")
    if "verify_signature" in lowered and "false" in lowered:
        probe_kinds.append("security-control-default")
    selected_path = selected.path.replace("\\", "/").lower() if selected else ""
    if (
        selected_path.startswith(".github/workflows/")
        and "${{" in lowered
    ):
        probe_kinds.append("github-actions-expression-shell")
    if (
        "unsafe_options" in lowered
        and "bare_unsafe_options" in lowered
        and "startswith" in lowered
    ):
        probe_kinds.append("git-option-normalization")
    if "replace(" in lowered and any(
        token in lowered for token in ("url", "location", "redirect")
    ):
        probe_kinds.append("url-normalization-redaction")
    if "model_dump" in lowered and any(
        token in lowered for token in ("setattr", "update")
    ):
        probe_kinds.append("serialization-exclusion-update")
    if "__ne__" in lowered or ("__eq__" in lowered and "not equal" in lowered):
        probe_kinds.append("equality-negation-contract")
    if "classmethod" in lowered and any(
        token in lowered for token in ("decorator", "fdefs_to_decorators")
    ):
        probe_kinds.append("decorator-order")
    if "_gc_cycle" in lowered or (
        "weakref" in lowered and "self_reference" in lowered
    ):
        probe_kinds.append("self-cycle-collection")
    if "validate_by_alias" in lowered and "validation_alias" in lowered:
        probe_kinds.append("alias-configuration-direction")
    if "def __getattr__" in lowered and any(
        token in lowered for token in ("deprecated", "warnings.warn", "deprecationwarning")
    ):
        probe_kinds.append("module-getattr-alias-bypass")
    if "_missing" in lowered and "default" in lowered:
        probe_kinds.append("sentinel-error-propagation")
    if "len(" in lowered and re.search(
        r"len\(\s*[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*", lowered,
    ):
        probe_kinds.append("nullable-length")
    loop_mutation = re.search(
        r"for\s+[a-z_][a-z0-9_]*\s+in\s+([a-z_][a-z0-9_]*)\s*:",
        lowered,
    )
    if loop_mutation and re.search(
        r"\b%s\.pop\s*\(" % re.escape(loop_mutation.group(1)), lowered,
    ):
        probe_kinds.append("dict-mutation-during-iteration")
    if (
        "as_tuple().exponent" in lowered
        and "isinstance(exponent, int)" not in lowered
        and any(operator in lowered for operator in (">=", "<=", " > ", " < "))
    ):
        probe_kinds.append("decimal-special-exponent")
    if re.search(r"memo\.add\(\s*exc_value\s*\)", lowered):
        probe_kinds.append("unhashable-exception-membership")
    if (
        re.search(r"(?m)^\s*with\s+os\.scandir\s*\(", semantic_text)
        and "except filenotfounderror" not in lowered
    ):
        probe_kinds.append("scandir-missing-directory")
    if re.search(r"(?m)^\s*if\s+_netrc\s*:\s*$", semantic_text):
        probe_kinds.append("empty-netrc-credentials")
    if (
        re.search(r"(?m)^\s*self\.refresh\(\)\s*$", semantic_text)
        and "self.stop()" not in lowered
        and not re.search(r"(?m)^\s*except\b", semantic_text)
    ):
        probe_kinds.append("exception-cleanup-state")
    if selected and "semantic_probe" in tools.names():
        for kind in probe_kinds[:3]:
            try:
                value = tools.invoke("semantic_probe", {"kind": kind})
                observations.append({
                    "step": 0, "tool": "semantic_probe", "ok": True,
                    "result": value,
                    "reason": "fixed semantic counterexample probe: " + kind,
                })
            except Exception as exc:
                observations.append({
                    "step": 0, "tool": "semantic_probe", "ok": False,
                    "error": str(exc)[:1000],
                })
    if repository_available and selected and "ast_analyze" in tools.names():
        try:
            value = tools.invoke("ast_analyze", {"path": selected.path})
            observations.append({
                "step": 0, "tool": "ast_analyze", "ok": True, "result": value,
                "reason": "evidence-first changed-file AST probe",
            })
        except Exception as exc:
            observations.append({
                "step": 0, "tool": "ast_analyze", "ok": False,
                "error": str(exc)[:1000],
            })
    return observations
