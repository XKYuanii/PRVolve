"""Run the same labelled PR cases with repository lessons disabled and enabled."""
import argparse
import hashlib
import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evoagent.agents.memory import MemoryManager  # noqa: E402
from evoagent.config import Settings  # noqa: E402
from evoagent.eval.agentic import ProductArmReviewer  # noqa: E402
from evoagent.eval.harness import EndToEndEvaluationHarness, load_jsonl  # noqa: E402
from evoagent.llm.client import JsonChatClient  # noqa: E402
from evoagent.store.postgres import create_store  # noqa: E402
from evoagent.store.sqlite import utc_now  # noqa: E402


METRICS = (
    "precision", "recall", "f1", "severity_accuracy", "high_risk_recall",
    "clean_accuracy", "execution_success_rate",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure whether verified repository lessons improve PR review.",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--token-budget", type=int, default=16000)
    parser.add_argument("--time-budget", type=int, default=120)
    args = parser.parse_args()
    if args.max_cases <= 0:
        parser.error("--max-cases must be positive")

    settings = Settings.from_env()
    config = settings.resolved_llm()
    if not config:
        parser.error("a configured LLM provider is required")
    cases = load_jsonl(args.dataset)[:args.max_cases]
    lesson_store = create_store(settings.database_url, settings.db_path)
    client = JsonChatClient(
        str(config["base_url"]), str(config["api_key"]), str(config["model"]),
        str(config["provider"]), settings.timeout_seconds,
        dict(config.get("headers") or {}),
    )
    baseline = ProductArmReviewer(
        "full-agentic", client, args.token_budget, args.time_budget,
        memory_manager=MemoryManager(lesson_store, enabled=False),
        tenant_id=settings.default_tenant_id,
    )
    candidate = ProductArmReviewer(
        "full-agentic", client, args.token_budget, args.time_budget,
        memory_manager=MemoryManager(
            lesson_store, enabled=True,
            recall_limit=settings.memory_recall_limit,
            working_ttl_seconds=settings.memory_working_ttl_seconds,
        ),
        tenant_id=settings.default_tenant_id,
    )
    harness = EndToEndEvaluationHarness()
    without_memory = harness.run(baseline, cases, "repository-lessons-off")
    with_memory = harness.run(candidate, cases, "repository-lessons-on")
    lesson_snapshot = []
    for repository in sorted({str(case["repository"]) for case in cases}):
        lesson_snapshot.extend({
            "id": item.get("id"), "repository": repository,
            "kind": item.get("kind"), "metadata": item.get("metadata"),
        } for item in lesson_store.list_agent_memories(
            settings.default_tenant_id, repository, ("semantic",), 200,
        ) if item.get("kind") == "repository_lesson")
    lesson_snapshot_json = json.dumps(
        lesson_snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "claim_scope": {
            "proves": "Paired review behavior with verified repository lessons off and on.",
            "does_not_prove": "Causality outside this fixed model, dataset and lesson snapshot.",
        },
        "model": {
            "provider": str(config["provider"]), "model": str(config["model"]),
            "token_budget": args.token_budget, "time_budget": args.time_budget,
        },
        "dataset_sha256": without_memory["dataset"]["sha256"],
        "lesson_snapshot": {
            "tenant_id": settings.default_tenant_id,
            "lessons": len(lesson_snapshot),
            "sha256": hashlib.sha256(lesson_snapshot_json.encode("utf-8")).hexdigest(),
        },
        "without_repository_lessons": without_memory,
        "with_repository_lessons": with_memory,
        "deltas": {
            key: round(
                with_memory["metrics"][key] - without_memory["metrics"][key], 4,
            )
            for key in METRICS
        },
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print("report:", output)
    print("deltas:", json.dumps(report["deltas"], sort_keys=True))


if __name__ == "__main__":
    main()
