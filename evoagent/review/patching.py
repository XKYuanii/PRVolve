"""LLM unified-patch generation with structural and sandbox verification."""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ..llm.client import JsonChatClient
from ..session.ledger import ExecutionLedger
from ..eval.verifier import RepairVerifier


PATCH_PROMPT = """You are the Fix Agent. Produce one minimal unified diff that fixes the supplied
verified findings without unrelated changes. Patch paths must be among allowed_paths. Preserve
behavior except for the defect. Return JSON only: {"patch":"--- a/path\n+++ b/path\n@@ ...",
"behavioral_claims":["..."],"related_tests":["..."]}. Do not use Markdown fences. If evidence is
insufficient, return {"patch":"","reason":"..."}."""


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: List[str]


@dataclass
class FilePatch:
    path: str
    hunks: List[Hunk]


HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_unified_patch(value: str, allowed_paths: List[str]) -> List[FilePatch]:
    lines = value.replace("\r\n", "\n").splitlines()
    patches: List[FilePatch] = []
    index = 0
    allowed = {item.replace("\\", "/") for item in allowed_paths}
    while index < len(lines):
        if not lines[index].startswith("--- "):
            if lines[index].strip():
                raise ValueError("patch contains content outside a file header")
            index += 1
            continue
        old_path = lines[index][4:].split("\t", 1)[0]
        index += 1
        if index >= len(lines) or not lines[index].startswith("+++ "):
            raise ValueError("patch is missing new-file header")
        new_path = lines[index][4:].split("\t", 1)[0]
        index += 1
        path = new_path[2:] if new_path.startswith("b/") else new_path
        old_normalized = old_path[2:] if old_path.startswith("a/") else old_path
        if path != old_normalized or path not in allowed:
            raise ValueError("patch path is not allowed: %s" % path)
        if path.startswith("/") or ".." in path.split("/"):
            raise ValueError("unsafe patch path")
        hunks = []
        while index < len(lines) and not lines[index].startswith("--- "):
            match = HUNK_HEADER.match(lines[index])
            if not match:
                if lines[index] == "\\ No newline at end of file":
                    index += 1
                    continue
                raise ValueError("invalid hunk header: %s" % lines[index][:100])
            old_start = int(match.group(1))
            old_count = int(match.group(2) or 1)
            new_start = int(match.group(3))
            new_count = int(match.group(4) or 1)
            index += 1
            body = []
            while index < len(lines):
                line = lines[index]
                if line.startswith("@@ ") or line.startswith("--- "):
                    break
                if not line.startswith((" ", "+", "-", "\\")):
                    raise ValueError("invalid unified patch line")
                if not line.startswith("\\"):
                    body.append(line)
                index += 1
            actual_old = sum(line[0] in {" ", "-"} for line in body)
            actual_new = sum(line[0] in {" ", "+"} for line in body)
            if actual_old != old_count or actual_new != new_count:
                raise ValueError("hunk line counts do not match header")
            hunks.append(Hunk(old_start, old_count, new_start, new_count, body))
        if not hunks:
            raise ValueError("file patch contains no hunks")
        patches.append(FilePatch(path, hunks))
    if not patches:
        raise ValueError("model returned no valid file patches")
    if len({item.path for item in patches}) != len(patches):
        raise ValueError("a file may appear only once in a generated patch")
    return patches


def apply_file_patch(content: str, patch: FilePatch) -> str:
    source = content.replace("\r\n", "\n").splitlines()
    output = []
    cursor = 0
    for hunk in patch.hunks:
        target = hunk.old_start - 1
        if target < cursor or target > len(source):
            raise ValueError("overlapping or out-of-range patch hunk")
        output.extend(source[cursor:target])
        cursor = target
        for line in hunk.lines:
            prefix, text = line[0], line[1:]
            if prefix in {" ", "-"}:
                if cursor >= len(source) or source[cursor] != text:
                    raise ValueError("patch context does not match source at line %d" % (cursor + 1))
                if prefix == " ":
                    output.append(text)
                cursor += 1
            elif prefix == "+":
                output.append(text)
    output.extend(source[cursor:])
    trailing = "\n" if content.endswith(("\n", "\r\n")) else ""
    return "\n".join(output) + trailing


class VerifiedPatchFixer:
    def __init__(self, client: JsonChatClient, verifier: RepairVerifier):
        self.client = client
        self.verifier = verifier

    def create_fix_commits(
        self, client, repository: str, pull_request: int, report: dict,
    ) -> dict:
        pull = client.get_pull_request(repository, pull_request)
        source_ref = pull["head"]["ref"]
        source_sha = pull["head"]["sha"]
        source_repository = pull["head"].get("repo", {}).get("full_name") or repository
        paths = sorted({
            str(item.get("path")) for item in report.get("findings", [])
            if item.get("path") and (item.get("gate") or {}).get("passed", True)
        })
        if not paths:
            return {"status": "suggestion-only", "branch": None, "commits": [],
                    "note": "No verified finding is eligible for patch generation."}
        originals = {}
        for path in paths:
            originals[path] = client.get_file(
                source_repository, path, source_ref
            )["decoded_content"]
        ledger = ExecutionLedger("fix-agent")
        generated = self.client.complete_json(
            "fix-agent", PATCH_PROMPT,
            json.dumps({
                "allowed_paths": paths, "findings": report.get("findings", []),
                "files": originals,
            }, ensure_ascii=False), ledger, 8000,
        )
        patch_text = str(generated.get("patch", ""))
        if not patch_text.strip():
            return {
                "status": "suggestion-only", "branch": None, "commits": [],
                "reason": str(generated.get("reason", "insufficient evidence"))[:1000],
                "execution": ledger.summary(),
                "note": "The Fix Agent did not produce an evidence-backed patch.",
            }
        patches = parse_unified_patch(patch_text, paths)
        files = dict(originals)
        for patch in patches:
            files[patch.path] = apply_file_patch(originals[patch.path], patch)
        changed = {path: content for path, content in files.items() if content != originals[path]}
        if not changed:
            raise ValueError("generated patch makes no change")
        structural = self.verifier.verify_contents(changed)
        if not structural["passed"]:
            return {
                "status": "blocked", "branch": None, "commits": [],
                "patch": patch_text, "verification": {"structural": structural},
                "execution": ledger.summary(),
                "note": "Patch failed AST/CST or compilation checks.",
            }
        if not self.verifier.test_command:
            return {
                "status": "suggestion-only", "branch": None, "commits": [],
                "patch": patch_text, "verification": {"structural": structural},
                "execution": ledger.summary(),
                "note": "No repository test command is configured; this is a suggestion, not a successful automatic fix.",
            }
        archive = client.download_archive(source_repository, source_sha)
        baseline = self.verifier.verify_archive(archive, {})
        patched = self.verifier.verify_archive(archive, changed)
        comparison = self.verifier.compare(baseline, patched)
        if not comparison["passed"]:
            return {
                "status": "blocked", "branch": None, "commits": [],
                "patch": patch_text,
                "verification": {
                    "structural": structural, "before": baseline,
                    "after": patched, "comparison": comparison,
                },
                "execution": ledger.summary(),
                "note": "Patch was blocked by before/after sandbox verification.",
            }
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        branch = "evoagent/fix-pr-%d-%s" % (pull_request, stamp)
        commit = client.create_atomic_commit(
            repository, branch, source_sha, changed,
            "fix: apply verified EvoAgent patch for PR #%d" % pull_request,
        )
        draft = client.create_draft_pull_request(
            repository, "fix: verified EvoAgent patch for #%d" % pull_request,
            branch, pull.get("base", {}).get("ref", "main"),
            "LLM-generated patch. AST/CST, compilation and configured tests passed in an isolated checkout. This PR is intentionally a draft.",
        )
        return {
            "status": "verified-draft", "branch": branch, "source_sha": source_sha,
            "commits": [{"sha": commit.get("sha"), "paths": sorted(changed)}],
            "draft_pull_request": {"number": draft.get("number"), "url": draft.get("html_url")},
            "patch": patch_text,
            "behavioral_claims": generated.get("behavioral_claims") or [],
            "related_tests": generated.get("related_tests") or [],
            "verification": {
                "structural": structural, "before": baseline,
                "after": patched, "comparison": comparison,
            },
            "execution": ledger.summary(),
            "note": "Verified patch was published only as a draft pull request.",
        }


class SuggestionOnlyFixer:
    def create_fix_commits(self, client, repository, pull_request, report):
        return {
            "status": "suggestion-only", "branch": None, "commits": [],
            "suggestions": [item.get("fix", "") for item in report.get("findings", [])],
            "note": "No model is configured. Suggestions are not described as an automatic fix.",
        }
