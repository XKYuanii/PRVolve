"""Prefetch risk-ranked repository context before a worker's first model call.

Workers that start blind burn their first turns on orientation. This runs the
read-only tools up front and hands the results in as observations, so the first
model call already carries the source, tests and call sites the assignment is
about.
"""
import builtins
import io
import keyword
import re
import tokenize


def code_identifiers(text):
    """Extract identifiers from partial Python hunks, never comment/string prose."""
    names = []
    # Diff fragments may have incomplete blocks or newer syntax. Tokenization
    # still gives useful names without requiring this interpreter to parse them.
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.NAME and not keyword.iskeyword(token.string):
                names.append(token.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return list(dict.fromkeys(names))


def repository_preflight(
    assignment, parsed, tools, repository_available=True,
):
    """Prefetch risk-ranked source context before a worker's first model call."""
    # A revision already receives the prior evidence selected by its evidence
    # mission.  Re-running the same automatic sweep spends tools without adding
    # facts; leave any missing proof to the Worker's own tool choice instead.
    if assignment.get("prior_worker_result"):
        return []
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
    targeted = []
    for target in assignment.get("evidence_targets") or []:
        if not isinstance(target, dict):
            continue
        path = str(target.get("path") or "")
        try:
            line = int(target.get("line") or 0)
        except (TypeError, ValueError):
            continue
        matches = [item for item in added if item.path == path]
        if not matches:
            continue
        exact = next((item for item in matches if item.line == line), None)
        selected_target = exact or min(matches, key=lambda item: abs(item.line - line))
        if selected_target not in targeted:
            targeted.append(selected_target)
    ignored = {
        "append", "format", "get", "items", "join", "strip", "replace",
        "self", "true", "false", "none", "return", "import", "from",
        "else", "with", "line", "value", "result", "object", "string",
        "info", "logger", "warning", "warn",
    } | set(dir(builtins)) | set(keyword.kwlist)
    observations = []
    selected_regions = []
    seen_regions = set()
    for item in targeted + [value for value in added if value not in targeted]:
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
    dependency_hits = []
    text = ""
    if selected:
        nearby = [
            item.content for item in sorted(added, key=lambda item: item.line)
            if item.path == selected.path and abs(item.line - selected.line) <= 12
        ]
        text = "\n".join(nearby or [selected.content])
        identifiers = code_identifiers(text)
        attributes = [
            name for name in re.findall(r"\.\s*([A-Za-z_][A-Za-z0-9_]*)", text)
            if name in identifiers
        ]
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
                    "search_repository", {
                        "query": query, "limit": 10,
                        "context_path": selected.path, "identifier": True,
                    }
                )
                observations.append({
                    "step": 0, "tool": "search_repository", "ok": True,
                    "result": value, "reason": "bounded identifier and local-contract search",
                })
                payload = value.get("output") if isinstance(value, dict) else None
                if isinstance(payload, list):
                    candidates = [
                        {**item, "searched_identifier": query} for item in payload
                        if isinstance(item, dict)
                        and item.get("path")
                        and not str(item.get("content", "")).lstrip().startswith(
                            ("import ", "from ", "#", "//")
                        )
                        and not any(
                            item.get("path") == region.path
                            and abs(int(item.get("line") or 0) - region.line) <= 25
                            for region in selected_regions
                        )
                    ]
                    dependency_hits.extend(candidates)
            except Exception as exc:
                observations.append({
                    "step": 0, "tool": "search_repository", "ok": False,
                    "error": str(exc)[:1000],
                })
    def dependency_priority(hit):
        content = str(hit.get("content") or "").lstrip()
        name = re.escape(hit["searched_identifier"])
        definition = bool(re.match(r"(?:async\s+)?(?:def|class)\s+%s\b" % name, content))
        call = bool(re.search(r"\b%s\s*\(" % name, content))
        test = any(part in str(hit["path"]).lower() for part in ("tests/", "/test", "test_"))
        return (0 if definition else 1 if call else 2, not test if call else False,
                hit["path"] != selected.path,
                str(hit["path"]), int(hit.get("line") or 0))

    dependency_hit = min(dependency_hits, key=dependency_priority) if dependency_hits else None
    if repository_available and dependency_hit and "read_file" in tools.names():
        try:
            cross_line = max(1, int(dependency_hit.get("line", 1)))
            value = tools.invoke("read_file", {
                "path": str(dependency_hit["path"]),
                "start_line": max(1, cross_line - 5),
                "end_line": cross_line + 65,
            })
            observations.append({
                "step": 0, "tool": "read_file", "ok": True,
                "result": value,
                "reason": "bounded producer, definition or test body for a changed identifier",
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
    evidence_text = " ".join(
        "%s %s" % (target.get("claim", ""), target.get("required_proof", ""))
        for target in assignment.get("evidence_targets") or []
        if isinstance(target, dict)
    ).lower()
    if evidence_text:
        if any(cue in evidence_text for cue in (
            "empty sequence", "empty string", "indexerror", "out of range",
            "empty index", "index into", "[-1]",
        )):
            probe_kinds.append("empty-sequence-index")
        if any(cue in evidence_text for cue in (
            "missing key", "missing mapping", "keyerror", "dictionary key",
        )):
            probe_kinds.append("missing-mapping-key")
        if any(cue in evidence_text for cue in (
            "truthiness", "truthy", "falsy", "empty value", "is not none",
            "explicit none", "explicitly supplied",
        )):
            probe_kinds.append("truthiness-vs-none")
        if any(cue in evidence_text for cue in (
            "json serial", "non-json", "range object", "not serializable",
        )):
            probe_kinds.append("json-serialization")
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
