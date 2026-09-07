"""Task evidence and verified repository lessons for review agents.

The store retains the historical scope names for compatibility. Product code
uses ``working`` for task-scoped recovery evidence and ``semantic`` for durable,
human- or test-verified repository lessons.
"""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..store.sqlite import utc_now


TOKEN = re.compile(r"[A-Za-z0-9_./:-]{2,}")
PRODUCT_SCOPES = {"working", "semantic"}
LEGACY_SCOPES = {"episodic", "procedural"}
# Legacy values remain readable so old databases and explicit integrations do
# not break. Current product flows write Working Evidence and Repository Lessons.
VALID_SCOPES = PRODUCT_SCOPES | LEGACY_SCOPES


def _tokens(value: str) -> set:
    return {item.lower() for item in TOKEN.findall(value)}


class MemoryManager:
    """Persist and retrieve bounded memories without hiding store side effects."""

    def __init__(
        self, store, enabled: bool = True, recall_limit: int = 3,
        working_ttl_seconds: int = 86400,
    ):
        self.store = store
        self.enabled = enabled
        self.recall_limit = max(1, recall_limit)
        self.working_ttl_seconds = max(60, working_ttl_seconds)

    def remember(
        self, tenant_id: str, repository: str, scope: str, kind: str,
        content: str, metadata: Optional[Dict[str, Any]] = None,
        task_id: str = "", agent: str = "", importance: float = 0.5,
        ttl_seconds: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled or not content.strip():
            return None
        if scope not in VALID_SCOPES:
            raise ValueError("unsupported memory scope: %s" % scope)
        importance = max(0.0, min(1.0, float(importance)))
        metadata = dict(metadata or {})
        normalized = content.strip()[:8000]
        identity = {
            "tenant": tenant_id, "repository": repository, "scope": scope,
            "kind": kind, "content": normalized, "metadata": metadata,
        }
        # Working observations belong to one task and one role.  Omitting this
        # ownership made identical tool output in a later PR collide with the
        # earlier row, leaving the new task unable to recall what it just wrote.
        if scope == "working":
            identity.update({"task_id": task_id, "agent": agent})
        elif scope == "semantic" and kind == "repository_lesson":
            # Provenance fields describe confirmations, not lesson identity.
            # Repeated confirmation of the same statement should not create an
            # ever-growing set of near-identical long-term memories.
            identity["metadata"] = {
                key: metadata.get(key)
                for key in ("lesson_kind", "category", "path_pattern", "symbol")
            }
        fingerprint = json.dumps(
            identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        memory_id = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
        ttl = self.working_ttl_seconds if scope == "working" and ttl_seconds is None else ttl_seconds
        expires_at = None
        if ttl:
            expires_at = (
                datetime.now(timezone.utc) + timedelta(seconds=max(1, int(ttl)))
            ).isoformat()
        record = {
            "id": memory_id, "tenant_id": tenant_id or "default",
            "repository": repository, "task_id": task_id, "agent": agent,
            "scope": scope, "kind": kind, "content": normalized,
            "keywords": sorted(_tokens(normalized) | _tokens(kind)),
            "metadata": metadata, "importance": importance,
            "created_at": utc_now(), "expires_at": expires_at,
        }
        return self.store.save_agent_memory(record)

    def recall(
        self, tenant_id: str, repository: str, query: str,
        scopes: Sequence[str] = ("semantic",),
        limit: Optional[int] = None, task_id: str = "",
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        purge = getattr(self.store, "purge_expired_agent_memories", None)
        if purge:
            purge()
        selected_scopes = tuple(scope for scope in scopes if scope in VALID_SCOPES)
        if not selected_scopes:
            return []
        candidates = self.store.list_agent_memories(
            tenant_id or "default", repository, selected_scopes, 200
        )
        if task_id:
            candidates = [
                item for item in candidates if str(item.get("task_id", "")) == str(task_id)
            ]
        query_tokens = _tokens(query)
        ranked = []
        fallbacks = []
        query_lower = query.lower()
        for index, item in enumerate(candidates):
            metadata = dict(item.get("metadata") or {})
            status = str(metadata.get("status", "active")).lower()
            if status == "refuted":
                continue
            memory_tokens = set(item.get("keywords") or []) | _tokens(item.get("content", ""))
            overlap = len(query_tokens.intersection(memory_tokens))
            coverage = overlap / max(1, len(query_tokens))
            specificity = overlap / max(1, len(memory_tokens))
            path_pattern = str(metadata.get("path_pattern", "")).lower()
            symbol = str(metadata.get("symbol", "")).lower()
            path_match = bool(path_pattern and path_pattern in query_lower)
            symbol_match = bool(symbol and symbol in query_lower)
            score = (
                coverage * 0.35 + specificity * 0.10
                + (0.25 if path_match else 0.0)
                + (0.20 if symbol_match else 0.0)
                + float(item.get("importance", 0.5)) * 0.08
                + (0.05 if metadata.get("verified_by") in {"human", "test"} else 0.0)
                + (-0.10 if status == "stale" else 0.0)
                + (0.05 / (index + 1))
            )
            value = dict(item)
            value["recall_score"] = round(score, 4)
            value["semantic_fallback"] = False
            if query_tokens and overlap == 0 and not path_match and not symbol_match:
                if item.get("scope") != "semantic":
                    continue
                verified = metadata.get("verified_by") in {"human", "test"} or (
                    item.get("kind") == "review_feedback"
                )
                if not verified:
                    continue
                repository_wide = bool(metadata.get("repository_wide")) or not (
                    path_pattern or symbol
                )
                if not repository_wide:
                    continue
                value["semantic_fallback"] = True
                fallbacks.append(value)
                continue
            ranked.append(value)
        size = max(1, limit or self.recall_limit)
        relevant = sorted(
            ranked,
            key=lambda item: (-item["recall_score"], item.get("created_at", "")),
        )[:size]
        if len(relevant) < size and fallbacks:
            relevant.append(sorted(
                fallbacks,
                key=lambda item: (-item["recall_score"], item.get("created_at", "")),
            )[0])
        return relevant

    def recall_working(
        self, tenant_id: str, repository: str, task_id: str, query: str = "",
        limit: Optional[int] = None, agent: str = "",
    ) -> List[Dict[str, Any]]:
        """Recall only the transient observations belonging to one review task."""
        if not task_id:
            return []
        values = self.recall(
            tenant_id, repository, query, scopes=("working",),
            # Filter by agent after the store query; request the bounded
            # candidate set first so another role's higher-importance entries
            # cannot hide this role's own observations.
            limit=200 if agent else limit,
            task_id=task_id,
        )
        selected = [
            item for item in values
            if not agent or str(item.get("agent", "")) == str(agent)
        ]
        return selected[:max(1, limit or self.recall_limit)]

    def remember_observation(
        self, tenant_id: str, repository: str, task_id: str, agent: str,
        observation: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Store a bounded, factual tool observation for later roles in this task."""
        result = observation.get("result")
        evidence_id = result.get("evidence_id", "") if isinstance(result, dict) else ""
        output = result.get("output") if isinstance(result, dict) else result
        try:
            rendered = json.dumps(output, ensure_ascii=False, default=str, separators=(",", ":"))
        except (TypeError, ValueError):
            rendered = str(output)
        content = "Agent %s tool %s at step %s: %s. Evidence: %s. Output: %s" % (
            agent, observation.get("tool", "unknown"), observation.get("step", 0),
            "ok" if observation.get("ok") else "failed", evidence_id,
            rendered[:4000] if observation.get("ok") else str(observation.get("error", ""))[:1000],
        )
        return self.remember(
            tenant_id, repository, "working", "tool_observation", content,
            {
                "tool": str(observation.get("tool", "")),
                "step": observation.get("step", 0), "ok": bool(observation.get("ok")),
                "evidence_id": evidence_id,
            }, task_id=task_id, agent=agent,
            importance=0.5 if observation.get("ok") else 0.3,
        )

    def remember_finding(
        self, tenant_id: str, repository: str, task_id: str,
        finding: Dict[str, Any], approved: bool, reasons: Iterable[str] = (),
    ) -> Optional[Dict[str, Any]]:
        """Legacy compatibility writer; production reviews do not call it."""
        decision = "approved" if approved else "rejected"
        content = (
            "%s finding %s at %s:%s. Evidence: %s. Explanation: %s. "
            "Fix: %s. Decision reasons: %s"
        ) % (
            decision, finding.get("rule_id", "unknown"), finding.get("path", ""),
            finding.get("line", 0), finding.get("evidence", ""),
            finding.get("explanation", ""), finding.get("fix", ""),
            "; ".join(str(item) for item in reasons),
        )
        return self.remember(
            tenant_id, repository, "episodic", "finding_%s" % decision,
            content, {"finding": finding, "approved": approved}, task_id=task_id,
            importance=0.8 if approved else 0.45,
        )

    def remember_feedback(
        self, tenant_id: str, repository: str, task_id: str, category: str,
        finding: Optional[Dict[str, Any]], note: str,
        review_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        finding = dict(finding or {})
        if category not in {"false_positive", "missed_issue", "bad_fix"}:
            return None
        lesson_kind = {
            "false_positive": "false_positive_rule",
            "missed_issue": "review_check",
            "bad_fix": "fix_constraint",
        }[category]
        path = str(finding.get("path", ""))
        symbol = str(finding.get("symbol", ""))
        content = "Verified repository lesson %s for %s at %s:%s. Note: %s" % (
            category, finding.get("rule_id", "task"), finding.get("path", ""),
            finding.get("line", 0), note,
        )
        return self.remember(
            tenant_id, repository, "semantic", "repository_lesson", content,
            {
                "lesson_kind": lesson_kind, "category": category,
                "finding": finding, "path_pattern": path, "symbol": symbol,
                "verified_by": "human", "status": "active",
                "repository_wide": not bool(path or symbol),
                "origin_task_id": task_id,
                "diff_sha256": str((review_context or {}).get("diff_sha256", "")),
                "source": str((review_context or {}).get("source", "")),
            }, task_id=task_id, importance=0.95,
        )

    def forget_working(self, task_id: str) -> int:
        if not self.enabled:
            return 0
        return self.store.delete_agent_memories(task_id=task_id, scope="working")

    def consolidate_task(
        self, tenant_id: str, repository: str, task_id: str,
        summary: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Legacy no-op that only releases transient task evidence."""
        if not self.enabled or not task_id:
            return None
        self.forget_working(task_id)
        return None
