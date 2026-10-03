from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from statistics import mean
from tempfile import mkdtemp
from typing import Any
import unicodedata

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    conversations = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(conversations, list):
        raise ValueError("Dataset must be a list of conversations")
    for conversation in conversations:
        if not isinstance(conversation, dict) or not all(isinstance(conversation.get(key), str) and conversation[key] for key in ("id", "user_id")):
            raise ValueError("Each conversation needs a non-empty id and user_id")
        if not isinstance(conversation.get("turns"), list) or not all(isinstance(turn, str) for turn in conversation["turns"]):
            raise ValueError("turns must be a list of strings")
        if not isinstance(conversation.get("recall_questions"), list):
            raise ValueError("recall_questions must be a list")
        for question in conversation["recall_questions"]:
            if not isinstance(question, dict) or not isinstance(question.get("question"), str) or not isinstance(question.get("expected_contains"), list) or not all(isinstance(item, str) and item for item in question["expected_contains"]):
                raise ValueError("Each recall question needs question text and expected_contains strings")
    return conversations


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _coverage(answer: str, expected: list[str]) -> float:
    if not expected:
        return 0.0
    normalized = _normalized(answer)
    return sum(_normalized(fact) in normalized for fact in expected) / len(expected)


def recall_points(answer: str, expected: list[str]) -> float:
    """0 for no matching facts, 0.5 for partial, 1 for all; empty targets score 0."""
    coverage = _coverage(answer, expected)
    return 1.0 if coverage == 1 else 0.5 if coverage > 0 else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Coverage times a brevity factor; a proxy, not a judge of correctness."""
    length = len(answer.strip())
    brevity = min(1.0, 600 / max(1, length))
    return _coverage(answer, expected) * (0.8 + 0.2 * brevity)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    users = {conversation["user_id"] for conversation in conversations}
    memory_size = getattr(agent, "memory_file_size", lambda user_id: 0)
    initial_bytes = sum(memory_size(user) for user in users)
    output_tokens = prompt_tokens = compactions = 0
    recalls, qualities = [], []
    for index, conversation in enumerate(conversations):
        user = conversation["user_id"]
        training_thread = f"train:{index}:{conversation['id']}"
        threads = [training_thread]
        initial_compactions = agent.compaction_count(training_thread)
        for message in conversation["turns"]:
            result = agent.reply(user, training_thread, message)
            output_tokens += result["agent_tokens"]
            prompt_tokens += result["prompt_tokens"]
        # Evaluate now: later conversations must not leak into this recall.
        for question_index, question in enumerate(conversation["recall_questions"]):
            recall_thread = f"recall:{index}:{question_index}:{conversation['id']}"
            threads.append(recall_thread)
            initial_compactions += agent.compaction_count(recall_thread)
            result = agent.reply(user, recall_thread, question["question"])
            output_tokens += result["agent_tokens"]
            prompt_tokens += result["prompt_tokens"]
            # expected_contains is given exclusively to the scorer.
            recalls.append(recall_points(result["answer"], question["expected_contains"]))
            qualities.append(heuristic_quality(result["answer"], question["expected_contains"]))
        compactions += sum(agent.compaction_count(thread) for thread in threads) - initial_compactions
    return BenchmarkRow(agent_name, output_tokens, prompt_tokens, mean(recalls) if recalls else 0.0,
                        mean(qualities) if qualities else 0.0,
                        sum(memory_size(user) for user in users) - initial_bytes, compactions)


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = ["Agent", "Agent tokens only", "Prompt tokens processed", "Cross-session recall", "Response quality", "Memory growth (bytes)", "Compactions"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        values = [row.agent_name, str(row.agent_tokens_only), str(row.prompt_tokens_processed),
                  f"{row.recall_score:.1%}", f"{row.response_quality:.1%}", str(row.memory_growth_bytes), str(row.compactions)]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare offline memory systems; --live explicitly enables API calls.")
    parser.add_argument("--live", action="store_true", help="Use the configured real provider (may incur API usage).")
    args = parser.parse_args()
    config = load_config(Path(__file__).resolve().parent.parent)
    output_root = config.state_dir / "benchmarks"
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(mkdtemp(prefix="run-", dir=output_root))
    suites = [("standard", "Standard Benchmark", "conversations.json"),
              ("stress", "Long-Context Stress Benchmark", "advanced_long_context.json")]
    results = {"mode": "live" if args.live else "offline", "token_accounting": "character estimates, including all dialogue and recall turns",
               "recall_rule": "0 / 0.5 / 1 for none / partial / all", "suites": {}}
    report = ["# Benchmark results", "", "Mode: " + results["mode"], "",
              "Tokens are character estimates; quality is an offline heuristic, not an LLM judge.", ""]
    for suite, title, filename in suites:
        suite_config = replace(config, state_dir=run_dir / suite, offline=not args.live)
        suite_config.state_dir.mkdir(parents=True, exist_ok=True)
        conversations = load_conversations(config.data_dir / filename)
        rows = [run_agent_benchmark("Baseline", BaselineAgent(suite_config, force_offline=not args.live), conversations, suite_config),
                run_agent_benchmark("Advanced", AdvancedAgent(suite_config, force_offline=not args.live), conversations, suite_config)]
        print("\n" + title + "\n" + format_rows(rows))
        results["suites"][suite] = [asdict(row) for row in rows]
        report.extend(["## " + title, "", format_rows(rows), ""])
    (run_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "results.md").write_text("\n".join(report), encoding="utf-8")
    print("\nTokens are estimates. Quality is a keyword/length heuristic.")
    print(f"Results and isolated profiles: {run_dir}")


if __name__ == "__main__":
    main()
