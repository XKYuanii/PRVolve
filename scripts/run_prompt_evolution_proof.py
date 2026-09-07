import argparse
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evoagent.evolution.proof import (  # noqa: E402
    generate_prompt_evolution_cases,
    run_prompt_evolution_proof,
    write_jsonl,
    write_report,
)
from evoagent.config import Settings  # noqa: E402
from evoagent.evolution.candidates import RootCauseEvolutionGenerator  # noqa: E402
from evoagent.llm.client import JsonChatClient  # noqa: E402
from evoagent.review.reviewers import OpenAICompatibleReviewer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an auditable feedback-driven prompt evolution replay."
    )
    parser.add_argument("--dataset", default="")
    parser.add_argument(
        "--output-dir", default=os.path.join("output", "prompt-evolution-proof")
    )
    parser.add_argument(
        "--reviewer", choices=("deterministic", "llm"), default="deterministic",
        help="Use the controlled proof reviewer or the configured production LLM reviewer.",
    )
    parser.add_argument(
        "--max-cases-per-split", type=int, default=None,
        help="Bound replay cost; LLM mode defaults to five cases per split.",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_path = args.dataset or os.path.join(
        args.output_dir, "prompt-evolution-cases.jsonl"
    )
    if not args.dataset:
        write_jsonl(generate_prompt_evolution_cases(), dataset_path)
    database_path = os.path.join(args.output_dir, "prompt-evolution-proof.db")
    if os.path.exists(database_path):
        raise SystemExit(
            "proof database already exists; choose a fresh --output-dir for an immutable run"
        )
    reviewer_factory = None
    candidate_generator = None
    reviewer_metadata = {"kind": "deterministic", "temperature": 0}
    if args.reviewer == "llm":
        settings = Settings.from_env()
        config = settings.resolved_llm()
        if not config:
            raise SystemExit("LLM reviewer requested but no provider is configured")
        client = JsonChatClient(
            str(config["base_url"]), str(config["api_key"]), str(config["model"]),
            str(config["provider"]), settings.timeout_seconds,
            dict(config.get("headers") or {}),
        )

        def reviewer_factory(prompt):
            return OpenAICompatibleReviewer(
                str(config["base_url"]), str(config["api_key"]),
                str(config["model"]), settings.timeout_seconds,
                system_prompt=prompt, provider=str(config["provider"]),
                extra_headers=dict(config.get("headers") or {}),
            )

        candidate_generator = RootCauseEvolutionGenerator(client)
        reviewer_metadata = {
            "kind": "external-llm", "provider": str(config["provider"]),
            "model": str(config["model"]), "temperature": 0,
            "timeout_seconds": settings.timeout_seconds,
        }
    max_cases = args.max_cases_per_split
    if max_cases is None:
        max_cases = 5 if args.reviewer == "llm" else 0
    if max_cases < 0:
        raise SystemExit("--max-cases-per-split must be non-negative")
    report = run_prompt_evolution_proof(
        dataset_path, database_path, reviewer_factory=reviewer_factory,
        candidate_generator=candidate_generator, reviewer_mode=args.reviewer,
        max_cases_per_split=max_cases, reviewer_metadata=reviewer_metadata,
    )
    paths = write_report(report, args.output_dir)
    print("decision:", report["evolution_run"]["decision"])
    print("run_id:", report["evolution_run"]["run_id"])
    print("json:", paths["json"])
    print("markdown:", paths["markdown"])


if __name__ == "__main__":
    main()
