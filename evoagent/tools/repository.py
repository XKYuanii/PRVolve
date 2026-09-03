"""Repository-level factual tools exposed to LLM agents through strict schemas."""
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import shlex
import time
import weakref
from typing import Any, Dict, Iterable, List, Optional, Set
from urllib.parse import urlsplit, urlunsplit

from ..core.diff_parser import ParsedDiff
from ..tools.registry import AgentTool, ToolRegistry
from ..session.ledger import ExecutionLedger


SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "dist", "build", ".venv", "venv"}
CONFIG_NAMES = {
    "package.json", "package-lock.json", "requirements.txt", "pyproject.toml",
    "poetry.lock", "Pipfile", "Dockerfile", "docker-compose.yml", ".github",
    "CODEOWNERS", "Cargo.toml", "go.mod", "pom.xml", "build.gradle",
}


def _evidence(tool: str, payload: Any) -> Dict[str, Any]:
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return {
        "evidence_id": "%s:%s" % (
            tool, hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]
        ),
        "tool": tool,
        "output": payload,
    }


class RepositoryToolSuite:
    def __init__(
        self, root: str, diff: str, parsed: ParsedDiff,
        ledger: Optional[ExecutionLedger] = None,
        test_command: str = "",
    ):
        candidate = os.path.abspath(root) if root else ""
        self.root = candidate if candidate and os.path.isdir(candidate) else ""
        self.diff = diff
        self.parsed = parsed
        self.ledger = ledger
        self.test_command = test_command

    @property
    def repository_available(self) -> bool:
        return bool(self.root)

    def _safe_path(self, relative: str) -> str:
        if not self.root:
            raise ValueError("repository checkout is unavailable for this review")
        relative = str(relative).replace("\\", "/").lstrip("/")
        target = os.path.abspath(os.path.join(self.root, relative))
        if target != self.root and not target.startswith(self.root + os.sep):
            raise ValueError("path escapes repository root")
        return target

    def _files(self) -> Iterable[str]:
        if not self.root:
            return []
        values = []
        for current, dirs, files in os.walk(self.root):
            dirs[:] = [item for item in dirs if item not in SKIP_DIRS]
            for name in files:
                path = os.path.join(current, name)
                try:
                    if os.path.getsize(path) <= 2_000_000:
                        values.append(os.path.relpath(path, self.root).replace("\\", "/"))
                except OSError:
                    continue
                if len(values) >= 20_000:
                    return values
        return values

    def list_repository(self, limit: int = 500) -> dict:
        files = list(self._files())[:max(1, min(int(limit), 2000))]
        return _evidence("list_repository", {
            "available": self.repository_available, "files": files,
            "truncated": len(files) >= min(int(limit), 2000),
        })

    def search_repository(self, query: str, limit: int = 50) -> dict:
        query = str(query).strip()
        if not query:
            raise ValueError("query is required")
        hits = []
        source_suffixes = (
            ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt",
            ".go", ".rs", ".rb", ".php", ".cs", ".c", ".cc", ".cpp", ".h",
        )
        # Search source before docs so bounded results expose code contracts.
        files = sorted(
            self._files(),
            key=lambda value: (not value.lower().endswith(source_suffixes), value),
        )
        for relative in files:
            try:
                with open(self._safe_path(relative), "r", encoding="utf-8", errors="replace") as handle:
                    for number, line in enumerate(handle, 1):
                        if query.lower() in line.lower():
                            hits.append({"path": relative, "line": number, "content": line.rstrip()[:500]})
                            if len(hits) >= max(1, min(int(limit), 200)):
                                return _evidence("search_repository", hits)
            except OSError:
                continue
        return _evidence("search_repository", hits)

    def search_diff(self, query: str, limit: int = 50) -> dict:
        query = str(query).strip().lower()
        if not query:
            raise ValueError("query is required")
        hits = [
            {"diff_line": number, "content": line[:500]}
            for number, line in enumerate(self.diff.splitlines(), 1)
            if query in line.lower()
        ][:max(1, min(int(limit), 200))]
        return _evidence("search_diff", hits)

    def read_file(self, path: str, start_line: int = 1, end_line: int = 240) -> dict:
        start = max(1, int(start_line))
        end = min(max(start, int(end_line)), start + 499)
        target = self._safe_path(path)
        with open(target, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
        payload = {
            "path": str(path), "start_line": start, "end_line": min(end, len(lines)),
            "content": "".join(lines[start - 1:end])[:40_000],
        }
        return _evidence("read_file", payload)

    def changed_line(self, path: str, line: int) -> dict:
        match = next((
            item for item in self.parsed.added_lines
            if item.path == str(path) and item.line == int(line)
        ), None)
        payload = (
            {"found": True, "path": match.path, "line": match.line, "content": match.content}
            if match else {"found": False, "path": path, "line": line}
        )
        return _evidence("changed_line", payload)

    def symbol(self, name: str) -> dict:
        name = str(name).strip()
        if not name:
            raise ValueError("symbol name is required")
        definitions, callers, callees = [], [], []
        for relative in self._files():
            if not relative.endswith(".py"):
                continue
            try:
                target = self._safe_path(relative)
                source = Path(target).read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source, filename=relative)
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                    definitions.append({"path": relative, "line": node.lineno, "kind": type(node).__name__})
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        for child in ast.walk(node):
                            if isinstance(child, ast.Call):
                                called = (
                                    child.func.id if isinstance(child.func, ast.Name)
                                    else child.func.attr if isinstance(child.func, ast.Attribute) else ""
                                )
                                if called:
                                    callees.append({"path": relative, "line": child.lineno, "symbol": called})
                if isinstance(node, ast.Call):
                    called = (
                        node.func.id if isinstance(node.func, ast.Name)
                        else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                    )
                    if called == name:
                        callers.append({"path": relative, "line": node.lineno})
            if len(definitions) + len(callers) > 300:
                break
        return _evidence("symbol", {
            "symbol": name, "definitions": definitions[:100],
            "callers": callers[:100], "callees": callees[:100],
        })

    def locate_tests(self, path: str = "", symbol: str = "") -> dict:
        tokens = {str(symbol).lower()} if symbol else set()
        if path:
            stem = Path(str(path)).stem.lower()
            tokens.update({stem, "test_" + stem})
        candidates = []
        for relative in self._files():
            lowered = relative.lower()
            if "test" not in lowered and "spec" not in lowered:
                continue
            score = sum(token and token in lowered for token in tokens)
            if score or not tokens:
                candidates.append({"path": relative, "relevance": score})
        candidates.sort(key=lambda item: (-item["relevance"], item["path"]))
        return _evidence("locate_tests", candidates[:100])

    def read_project_controls(self) -> dict:
        selected = []
        for relative in self._files():
            parts = set(relative.split("/"))
            if Path(relative).name in CONFIG_NAMES or parts.intersection({".github", "config", "permissions"}):
                try:
                    content = Path(self._safe_path(relative)).read_text(
                        encoding="utf-8", errors="replace"
                    )[:20_000]
                except OSError:
                    continue
                selected.append({"path": relative, "content": content})
                if len(selected) >= 40:
                    break
        return _evidence("read_project_controls", selected)

    def ast_analyze(self, path: str) -> dict:
        target = self._safe_path(path)
        source = Path(target).read_text(encoding="utf-8", errors="replace")
        if str(path).endswith(".py"):
            try:
                tree = ast.parse(source, filename=str(path))
            except SyntaxError as exc:
                return _evidence("ast_analyze", {
                    "path": path, "valid": False, "error": str(exc),
                })
            nodes = {}
            dangerous = []
            for node in ast.walk(tree):
                kind = type(node).__name__
                nodes[kind] = nodes.get(kind, 0) + 1
                if isinstance(node, ast.Call):
                    called = (
                        node.func.id if isinstance(node.func, ast.Name)
                        else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                    )
                    if called in {"eval", "exec", "system", "Popen", "run", "loads"}:
                        dangerous.append({"symbol": called, "line": node.lineno})
            return _evidence("ast_analyze", {
                "path": path, "valid": True, "node_counts": nodes,
                "dangerous_calls": dangerous,
            })
        return _evidence("ast_analyze", {
            "path": path, "valid": None,
            "note": "Tree-sitter is not installed; non-Python AST analysis is unavailable.",
        })

    @staticmethod
    def semantic_probe(kind: str) -> dict:
        """Run a fixed, side-effect-free language or data-transformation probe."""
        kind = str(kind).strip().lower()
        if kind == "path-containment":
            base = os.path.abspath(os.path.join(os.sep, "workspace", "repository"))
            relative = os.path.join("..", "..", "sensitive.txt")
            resolved = os.path.abspath(os.path.join(base, relative))
            try:
                contained = os.path.commonpath([base, resolved]) == base
            except ValueError:
                contained = False
            return _evidence("semantic_probe", {
                "kind": kind,
                "operation": "join-base-with-parent-segments-then-normalize",
                "base": base,
                "relative_input": relative,
                "resolved": resolved,
                "contained_under_base": contained,
                "parent_segments_escape_base": not contained,
                "network_used": False,
                "filesystem_read": False,
                "arbitrary_code_executed": False,
            })
        if kind == "security-control-default":
            configuration = {}
            default = False
            effective = configuration.get("verify_signature", default)
            return _evidence("semantic_probe", {
                "kind": kind,
                "operation": "missing-configuration-uses-security-control-default",
                "configuration": configuration,
                "default": default,
                "effective_verify_signature": effective,
                "verification_branch_entered": bool(effective),
                "security_control_disabled_by_default": effective is False,
                "network_used": False,
                "arbitrary_code_executed": False,
            })
        if kind == "github-actions-expression-shell":
            attacker_value = 'target"; echo INJECTED; #'
            direct_command = (
                'pytest --chunked-sites "' + attacker_value + '"'
            )
            environment_command = 'pytest --chunked-sites "$CHANGED_TARGETS"'
            return _evidence("semantic_probe", {
                "kind": kind,
                "operation": "compare-expression-interpolation-with-environment-indirection",
                "attacker_controlled_value": attacker_value,
                "direct_expression_command": direct_command,
                "environment_reference_command": environment_command,
                "direct_command_contains_attacker_text": (
                    attacker_value in direct_command
                ),
                "shell_metacharacters_reach_direct_command": all(
                    token in direct_command for token in ('"', ';', '#')
                ),
                "environment_reference_keeps_value_out_of_command_text": (
                    attacker_value not in environment_command
                ),
                "network_used": False,
                "arbitrary_code_executed": False,
            })
        if kind == "git-option-normalization":
            supplied_kwarg = "upload_pack"
            unsafe_option = "--upload-pack"
            raw_name = supplied_kwarg.lstrip("-").split("=", 1)[0]
            canonical_name = raw_name.replace("_", "-")
            unsafe_name = unsafe_option.lstrip("-").split("=", 1)[0]
            emitted_flag = "--%s=/tmp/helper" % canonical_name
            return _evidence("semantic_probe", {
                "kind": kind,
                "operation": "compare-raw-kwarg-check-with-emitted-git-option",
                "supplied_kwarg": supplied_kwarg,
                "unsafe_option": unsafe_option,
                "raw_name": raw_name,
                "canonical_name": canonical_name,
                "raw_check_blocks": raw_name == unsafe_name,
                "canonical_check_blocks": canonical_name == unsafe_name,
                "emitted_flag": emitted_flag,
                "dangerous_flag_emitted_after_raw_check": (
                    raw_name != unsafe_name
                    and emitted_flag.startswith(unsafe_option + "=")
                ),
                "network_used": False,
                "arbitrary_code_executed": False,
            })
        if kind == "url-normalization-redaction":
            original = "http://user:s%7Eecret@example.invalid/objects.inv"
            unreserved = frozenset(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
            )

            def normalize_match(match):
                character = chr(int(match.group(0)[1:], 16))
                return character if character in unreserved else match.group(0).upper()

            normalized = re.sub(r"%[0-9A-Fa-f]{2}", normalize_match, original)
            parts = urlsplit(original)
            safe = urlunsplit((parts.scheme, parts.hostname or "", parts.path,
                               parts.query, parts.fragment))
            replaced = normalized.replace(original, safe)
            remaining = urlsplit(replaced)
            payload = {
                "kind": kind,
                "operation": "normalize-unreserved-then-exact-string-replace",
                "original": original,
                "normalized": normalized,
                "redacted_original": safe,
                "replacement_result": replaced,
                "exact_original_still_matches": original in normalized,
                "credentials_remaining": bool(
                    remaining.username is not None or remaining.password is not None
                ),
                "network_used": False,
                "arbitrary_code_executed": False,
            }
            return _evidence("semantic_probe", payload)
        if kind == "tri-state-boolean":
            return _evidence("semantic_probe", {
                "kind": kind,
                "values": [
                    {"value": repr(value), "not_value": not value,
                     "is_false": value is False}
                    for value in (None, False, True)
                ],
                "network_used": False,
                "arbitrary_code_executed": False,
            })
        if kind == "nullable-length":
            value = None
            error_type = ""
            try:
                len(value)
            except TypeError as exc:
                error_type = type(exc).__name__
            return _evidence("semantic_probe", {
                "kind": kind,
                "operation": "apply-len-to-an-optional-attribute-value",
                "value": None,
                "raises_when_value_is_none": error_type == "TypeError",
                "error_type": error_type,
                "network_used": False,
                "filesystem_read": False,
                "arbitrary_code_executed": False,
            })
        if kind == "dict-mutation-during-iteration":
            values = {"remaining": "value"}
            error_type = ""
            try:
                for key in values:
                    values.pop(key)
            except RuntimeError as exc:
                error_type = type(exc).__name__
            return _evidence("semantic_probe", {
                "kind": kind,
                "operation": "pop-current-key-while-iterating-dict",
                "dict_size_change_raises": error_type == "RuntimeError",
                "error_type": error_type,
                "requires_resolution": True,
                "resolution_question": (
                    "Python raises RuntimeError when a dict changes size during iteration. "
                    "Show that the added loop cannot pop from the iterated dict, or report it."
                ),
                "network_used": False,
                "filesystem_read": False,
                "arbitrary_code_executed": False,
            })
        if kind == "decimal-special-exponent":
            from decimal import Decimal

            values = []
            for literal in ("NaN", "Infinity"):
                exponent = Decimal(literal).as_tuple().exponent
                error_type = ""
                try:
                    exponent >= 0
                except TypeError as exc:
                    error_type = type(exc).__name__
                values.append({
                    "literal": literal,
                    "exponent": str(exponent),
                    "exponent_type": type(exponent).__name__,
                    "comparison_with_zero_raises": error_type == "TypeError",
                    "error_type": error_type,
                })
            return _evidence("semantic_probe", {
                "kind": kind,
                "operation": "compare-special-decimal-exponents-with-zero",
                "values": values,
                "requires_resolution": True,
                "resolution_question": (
                    "Decimal NaN and Infinity expose non-integer exponent sentinels that "
                    "raise TypeError when compared with zero. Show they are rejected before "
                    "the added comparison, or report the changed line."
                ),
                "network_used": False,
                "filesystem_read": False,
                "arbitrary_code_executed": False,
            })
        if kind == "unhashable-exception-membership":
            class UnhashableException(Exception):
                __hash__ = None

            value = UnhashableException("probe")
            error_type = ""
            try:
                {value}
            except TypeError as exc:
                error_type = type(exc).__name__
            return _evidence("semantic_probe", {
                "kind": kind,
                "operation": "insert-unhashable-exception-in-set",
                "exception_hash_is_none": type(value).__hash__ is None,
                "set_insertion_raises": error_type == "TypeError",
                "error_type": error_type,
                "requires_resolution": True,
                "resolution_question": (
                    "Python exception subclasses may disable hashing, and inserting such an "
                    "instance into a set raises TypeError. Show the added set operation only "
                    "receives hashable values, or report it."
                ),
                "network_used": False,
                "filesystem_read": False,
                "arbitrary_code_executed": False,
            })
        if kind == "scandir-missing-directory":
            def missing_scandir(_path):
                raise FileNotFoundError("probe directory disappeared")

            error_type = ""
            try:
                with missing_scandir("missing") as entries:
                    list(entries)
            except FileNotFoundError as exc:
                error_type = type(exc).__name__
            return _evidence("semantic_probe", {
                "kind": kind,
                "operation": "propagate-scandir-open-error-without-handler",
                "scandir_open_raises": error_type == "FileNotFoundError",
                "error_type": error_type,
                "requires_resolution": True,
                "resolution_question": (
                    "Opening a directory can raise FileNotFoundError before the context "
                    "manager is entered. Show that the added direct os.scandir call catches "
                    "that race, or report its exact changed line."
                ),
                "network_used": False,
                "filesystem_read": False,
                "arbitrary_code_executed": False,
            })
        if kind == "empty-netrc-credentials":
            credentials = ("", "", "")
            return _evidence("semantic_probe", {
                "kind": kind,
                "operation": "compare-tuple-truthiness-with-field-truthiness",
                "credentials": list(credentials),
                "tuple_is_truthy": bool(credentials),
                "any_field_is_truthy": any(credentials),
                "blank_credentials_pass_tuple_truthiness": (
                    bool(credentials) and not any(credentials)
                ),
                "requires_resolution": True,
                "resolution_question": (
                    "A non-empty tuple of blank netrc fields is truthy even though every "
                    "credential field is empty. Show that the added tuple-only check rejects "
                    "this value, or report its exact changed line."
                ),
                "network_used": False,
                "filesystem_read": False,
                "arbitrary_code_executed": False,
            })
        if kind == "exception-cleanup-state":
            state = {"started": False, "redirect_enabled": False,
                     "render_hook_installed": False}

            def start_without_cleanup():
                state["started"] = True
                state["redirect_enabled"] = True
                state["render_hook_installed"] = True
                raise RuntimeError("initial refresh failed")

            error_type = ""
            try:
                start_without_cleanup()
            except RuntimeError as exc:
                error_type = type(exc).__name__
            return _evidence("semantic_probe", {
                "kind": kind,
                "operation": "raise-after-installing-state-without-exception-cleanup",
                "error_type": error_type,
                "state_after_exception": dict(state),
                "state_remains_installed": all(state.values()),
                "requires_resolution": True,
                "resolution_question": (
                    "An exception raised after started state, stream redirection and a "
                    "render hook are installed leaves all three active without an exception "
                    "cleanup path. Show that the added refresh call cannot raise or that the "
                    "state is cleaned up, or report its exact changed line."
                ),
                "network_used": False,
                "filesystem_read": False,
                "arbitrary_code_executed": False,
            })
        if kind == "sentinel-error-propagation":
            missing = object()
            conversion_failed = False
            try:
                result = int("not-an-integer")
            except (TypeError, ValueError):
                conversion_failed = True
                result = missing
            return _evidence("semantic_probe", {
                "kind": kind,
                "operation": "conversion-error-falls-through-to-missing-sentinel",
                "conversion_failed": conversion_failed,
                "returned_missing_sentinel": result is missing,
                "exception_propagated": not conversion_failed,
                "network_used": False,
                "filesystem_read": False,
                "arbitrary_code_executed": False,
            })
        if kind == "serialization-exclusion-update":
            current = {"id": "1", "excluded_field": "old"}
            incoming_object = {"id": "1", "excluded_field": "new"}
            serialized = {"id": incoming_object["id"]}
            update_payload = {
                key: value for key, value in serialized.items() if key != "id"
            }
            updated = dict(current)
            updated.update(update_payload)
            return _evidence("semantic_probe", {
                "kind": kind,
                "operation": "serialize-omitting-field-then-update-excluding-id",
                "current": current,
                "incoming_object": incoming_object,
                "serialized": serialized,
                "update_payload": update_payload,
                "updated": updated,
                "excluded_field_update_lost": (
                    updated["excluded_field"] != incoming_object["excluded_field"]
                ),
                "arbitrary_code_executed": False,
            })
        if kind == "equality-negation-contract":
            left = {"plain": "same", "spans": ["red"]}
            right = {"plain": "same", "spans": ["blue"]}
            equal = left == right
            not_equal_plain_only = left["plain"] != right["plain"]
            return _evidence("semantic_probe", {
                "kind": kind,
                "left": left,
                "right": right,
                "eq_full_state": equal,
                "ne_plain_only": not_equal_plain_only,
                "logical_inverse_of_eq": not equal,
                "contract_violated": not_equal_plain_only == equal,
                "arbitrary_code_executed": False,
            })
        if kind == "decorator-order":
            function = "function"
            correct = ("outer", ("classmethod", function))
            reordered = ("classmethod", ("outer", function))
            return _evidence("semantic_probe", {
                "kind": kind,
                "source_order": ["outer", "classmethod"],
                "python_application": correct,
                "descriptor_forced_outermost": reordered,
                "same_result": correct == reordered,
                "arbitrary_code_executed": False,
            })
        if kind == "self-cycle-collection":
            class CycleProbe:
                pass

            value = CycleProbe()
            reference = weakref.ref(value)
            value.self_reference = value
            del value
            alive_before_collection = reference() is not None
            gc.collect()
            alive_after_collection = reference() is not None
            return _evidence("semantic_probe", {
                "kind": kind,
                "alive_after_last_external_reference_deleted": alive_before_collection,
                "alive_after_cyclic_gc": alive_after_collection,
                "collection_delayed_until_cyclic_gc": (
                    alive_before_collection and not alive_after_collection
                ),
                "arbitrary_code_executed": False,
            })
        if kind == "alias-configuration-direction":
            return _evidence("semantic_probe", {
                "kind": kind,
                "truth_table": [
                    {
                        "validate_by_alias": enabled,
                        "is_false_selected_key": (
                            "field_name" if enabled is False else "validation_alias"
                        ),
                        "falsy_selected_key": (
                            "field_name" if not enabled else "validation_alias"
                        ),
                        "conditions_diverge": (enabled is False) != (not enabled),
                    }
                    for enabled in (None, False, True)
                ],
                "requires_resolution": True,
                "resolution_question": (
                    "The proposed `is False` condition diverges from a falsy check when the "
                    "configuration value is None. Show repository evidence that None is "
                    "unreachable here, or report the changed condition as a finding."
                ),
                "network_used": False,
                "filesystem_read": False,
                "arbitrary_code_executed": False,
            })
        if kind == "module-getattr-alias-bypass":
            module_globals = {"legacy_name": "existing-function"}
            requested = "legacy_name"
            fallback_invoked = requested not in module_globals
            return _evidence("semantic_probe", {
                "kind": kind,
                "operation": "resolve-existing-module-global-before-module-getattr",
                "requested_name": requested,
                "name_exists_in_module_globals": requested in module_globals,
                "module_getattr_invoked": fallback_invoked,
                "deprecation_warning_path_reached": fallback_invoked,
                "requires_resolution": True,
                "resolution_question": (
                    "Python resolves an existing module global without calling module __getattr__. "
                    "Show that each deprecated public name is absent from that module, or report "
                    "the warning bypass as a finding."
                ),
                "network_used": False,
                "filesystem_read": False,
                "arbitrary_code_executed": False,
            })
        raise ValueError(
            "unsupported semantic probe kind"
        )

    def git_context(self, path: str, line: int = 1) -> dict:
        target = self._safe_path(path)
        if not os.path.isdir(os.path.join(self.root, ".git")) or not shutil.which("git"):
            return _evidence("git_context", {"available": False, "path": path})
        commands = [
            ["git", "log", "-n", "8", "--format=%h%x09%ad%x09%s", "--date=short", "--", str(path)],
            ["git", "blame", "-L", "%d,%d" % (max(1, int(line) - 3), max(1, int(line) + 3)), "--", str(path)],
        ]
        outputs = []
        for command in commands:
            result = subprocess.run(
                command, cwd=self.root, capture_output=True, text=True,
                timeout=20, check=False,
            )
            outputs.append({
                "command": command[1], "returncode": result.returncode,
                "output": (result.stdout + result.stderr)[-12_000:],
            })
        return _evidence("git_context", {"available": True, "path": path, "results": outputs})

    def run_scanners(self) -> dict:
        """Run installed, fixed-command analyzers. User input never becomes a command."""
        if not self.root:
            return _evidence("run_scanners", {"available": False, "runs": []})
        specs = [
            ("semgrep", ["semgrep", "scan", "--config", "auto", "--json", "--quiet", "."]),
            ("bandit", ["bandit", "-r", ".", "-f", "json", "-q"]),
            ("eslint", ["eslint", ".", "--format", "json"]),
            ("mypy", ["mypy", ".", "--no-error-summary"]),
            ("pyright", ["pyright", "--outputjson"]),
        ]
        runs = []
        for name, command in specs:
            if not shutil.which(command[0]):
                runs.append({"name": name, "available": False})
                continue
            try:
                result = subprocess.run(
                    command, cwd=self.root, capture_output=True, text=True,
                    timeout=90, check=False,
                    env={key: value for key, value in os.environ.items() if key in {
                        "PATH", "SYSTEMROOT", "WINDIR", "LANG", "LC_ALL", "TMP", "TEMP"
                    }},
                )
                runs.append({
                    "name": name, "available": True, "returncode": result.returncode,
                    "output": (result.stdout + result.stderr)[-50_000:],
                })
            except subprocess.TimeoutExpired:
                runs.append({"name": name, "available": True, "timed_out": True})
        return _evidence("run_scanners", {"available": True, "runs": runs})

    def run_repository_checks(self, kind: str = "compile") -> dict:
        """Run fixed checks in an isolated repository copy with a restricted environment."""
        kind = str(kind).strip().lower()
        if kind not in {"compile", "tests"}:
            raise ValueError("kind must be compile or tests")
        if not self.root:
            return _evidence("test", {"available": False, "kind": kind})
        if kind == "tests" and not self.test_command:
            return _evidence("test", {
                "available": False, "kind": kind,
                "reason": "no administrator-configured test command",
            })
        with tempfile.TemporaryDirectory(prefix="evoagent-review-check-") as temp:
            checkout = os.path.join(temp, "checkout")
            shutil.copytree(
                self.root, checkout,
                ignore=shutil.ignore_patterns(*SKIP_DIRS),
            )
            command = (
                [sys.executable, "-m", "compileall", "-q", "."]
                if kind == "compile" else
                shlex.split(self.test_command, posix=os.name != "nt")
            )
            env = {
                key: value for key, value in os.environ.items()
                if key in {"PATH", "SYSTEMROOT", "WINDIR", "LANG", "LC_ALL", "TMP", "TEMP"}
            }
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            try:
                result = subprocess.run(
                    command, cwd=checkout, env=env, capture_output=True, text=True,
                    timeout=120, check=False,
                )
                payload = {
                    "available": True, "kind": kind,
                    "isolation": "temporary-copy+restricted-environment",
                    "network_disabled": False,
                    "returncode": result.returncode,
                    "passed": result.returncode == 0,
                    "output": (result.stdout + result.stderr)[-40_000:],
                }
            except subprocess.TimeoutExpired:
                payload = {
                    "available": True, "kind": kind,
                    "isolation": "temporary-copy+restricted-environment",
                    "network_disabled": False, "passed": False, "timed_out": True,
                }
        return _evidence("test", payload)

    def registry(self, role: str, allowed: Optional[Set[str]] = None) -> ToolRegistry:
        specs = {
            "list_repository": (
                "List repository files.",
                {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 2000}}, "additionalProperties": False},
                self.list_repository,
            ),
            "search_repository": (
                "Full repository text search.",
                {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 200}}, "required": ["query"], "additionalProperties": False},
                self.search_repository,
            ),
            "search_diff": (
                "Search the unified diff.",
                {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 200}}, "required": ["query"], "additionalProperties": False},
                self.search_diff,
            ),
            "read_file": (
                "Read a bounded range from a repository file.",
                {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer", "minimum": 1}, "end_line": {"type": "integer", "minimum": 1}}, "required": ["path"], "additionalProperties": False},
                self.read_file,
            ),
            "changed_line": (
                "Read one added line from the diff.",
                {"type": "object", "properties": {"path": {"type": "string"}, "line": {"type": "integer", "minimum": 1}}, "required": ["path", "line"], "additionalProperties": False},
                self.changed_line,
            ),
            "symbol": (
                "Find a symbol definition, callers and callees.",
                {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"], "additionalProperties": False},
                self.symbol,
            ),
            "locate_tests": (
                "Locate tests related to a path or symbol.",
                {"type": "object", "properties": {"path": {"type": "string"}, "symbol": {"type": "string"}}, "additionalProperties": False},
                self.locate_tests,
            ),
            "read_project_controls": (
                "Read dependency, configuration and permission files.",
                {"type": "object", "properties": {}, "additionalProperties": False},
                self.read_project_controls,
            ),
            "ast_analyze": (
                "Parse source and return AST facts.",
                {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False},
                self.ast_analyze,
            ),
            "semantic_probe": (
                "Run a fixed side-effect-free semantic counterexample probe; no arbitrary code or network.",
                {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": [
                                "path-containment",
                                "security-control-default",
                                "github-actions-expression-shell",
                                "git-option-normalization",
                                "url-normalization-redaction",
                                "tri-state-boolean",
                                "nullable-length",
                                "dict-mutation-during-iteration",
                                "decimal-special-exponent",
                                "unhashable-exception-membership",
                                "scandir-missing-directory",
                                "empty-netrc-credentials",
                                "sentinel-error-propagation",
                                "serialization-exclusion-update",
                                "equality-negation-contract",
                                "decorator-order",
                                "self-cycle-collection",
                                "alias-configuration-direction",
                                "module-getattr-alias-bypass",
                            ],
                        }
                    },
                    "required": ["kind"],
                    "additionalProperties": False,
                },
                self.semantic_probe,
            ),
            "git_context": (
                "Read nearby Git history and blame context.",
                {"type": "object", "properties": {"path": {"type": "string"}, "line": {"type": "integer", "minimum": 1}}, "required": ["path"], "additionalProperties": False},
                self.git_context,
            ),
            "run_scanners": (
                "Run installed Semgrep, Bandit, ESLint and type checkers with fixed arguments.",
                {"type": "object", "properties": {}, "additionalProperties": False},
                self.run_scanners,
            ),
            "run_repository_checks": (
                "Compile or run the administrator-configured tests in an isolated checkout copy.",
                {"type": "object", "properties": {"kind": {"type": "string"}}, "required": ["kind"], "additionalProperties": False},
                self.run_repository_checks,
            ),
        }
        selected = allowed or set(specs)
        tools = []
        for name in sorted(selected.intersection(specs)):
            description, schema, handler = specs[name]

            def wrapped(_handler=handler, _name=name, **arguments):
                started = time.monotonic()
                try:
                    value = _handler(**arguments)
                    if self.ledger:
                        self.ledger.record_tool(
                            role, _name, arguments, True,
                            int((time.monotonic() - started) * 1000), value,
                        )
                    return value
                except Exception as exc:
                    if self.ledger:
                        self.ledger.record_tool(
                            role, _name, arguments, False,
                            int((time.monotonic() - started) * 1000), error=str(exc),
                        )
                    raise
            tools.append(AgentTool(name, description, schema, wrapped))
        return ToolRegistry(tools)
