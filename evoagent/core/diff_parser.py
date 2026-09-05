import re
from dataclasses import dataclass
from typing import List

from ..core.models import ChangedLine


@dataclass
class ParsedDiff:
    files: List[str]
    added_lines: List[ChangedLine]


HUNK = re.compile(r"^@@ -(?:\d+)(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def parse_unified_diff(diff: str) -> ParsedDiff:
    files: List[str] = []
    added: List[ChangedLine] = []
    current_path = ""
    new_line = 0
    in_hunk = False

    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            current_path = raw[4:].strip()
            if current_path.startswith("b/"):
                current_path = current_path[2:]
            if current_path != "/dev/null" and current_path not in files:
                files.append(current_path)
            in_hunk = False
            continue
        match = HUNK.match(raw)
        if match:
            new_line = int(match.group(1))
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            added.append(ChangedLine(current_path or "unknown", new_line, raw[1:]))
            new_line += 1
        elif raw.startswith("-") and not raw.startswith("---"):
            continue
        elif raw.startswith("\\ No newline"):
            continue
        else:
            new_line += 1

    return ParsedDiff(files=files, added_lines=added)


def changed_block(diff: str, path: str, line: int) -> dict:
    """Return both sides of the contiguous edit containing a new-file line.

    Keep replacements as blocks: adjacent +/- lines need not be one-to-one,
    and some imported diffs put additions before deletions.
    """
    current_path, new_line, start = "", None, 0
    before, after = [], []
    for raw in [*diff.splitlines(), ""]:
        if raw.startswith("\\ No newline"):
            continue
        edit = new_line is not None and raw[:1] in {"+", "-"} and not raw.startswith(
            ("--- ", "+++ ")
        )
        if not edit and (before or after):
            if current_path == path and start <= line < start + len(after):
                old, new = "\n".join(before), "\n".join(after)
                return {
                    "before": old[:4000], "after": new[:4000],
                    "complete": len(old) <= 4000 and len(new) <= 4000,
                }
            before, after = [], []
        if raw.startswith("+++ "):
            current_path = raw[4:].split("\t", 1)[0]
            if current_path.startswith("b/"):
                current_path = current_path[2:]
            new_line = None
        elif raw.startswith(("diff --git ", "--- ")):
            new_line = None
        elif HUNK.match(raw):
            new_line = int(HUNK.match(raw).group(1))
        elif edit:
            if not before and not after:
                start = new_line
            (after if raw.startswith("+") else before).append(raw[1:])
            if raw.startswith("+"):
                new_line += 1
        elif new_line is not None and raw.startswith(" "):
            new_line += 1
    return {}

