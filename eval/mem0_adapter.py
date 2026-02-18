#!/usr/bin/env python3
"""AgentLife Benchmark — Mem0 Adapter.

Integrates Mem0 (open-source self-hosted) with the AgentLife benchmark.
Mem0 runs on the host machine via Python API — no VM needed.

Requires:
    pip install mem0ai

Usage:
    # Standalone test
    python3 mem0_adapter.py --test

    # Used by vm_benchmark.py automatically
"""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_DIR = Path(__file__).resolve().parent
_WORKSPACE = Path(os.environ.get("CLAWDBOT_WORKSPACE", Path.home() / "clawd"))
_RUNNER_DIR = _WORKSPACE / "memory-stress-test" / "runner"

sys.path.insert(0, str(_DIR))
if str(_RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(_RUNNER_DIR))

from claude_backend import call_claude
from dataset import SESSION_DATES
from injector import transcript_to_messages, count_tokens
from run_production_benchmark import _judge, _judge_tier5, _get_api_key

# ---------------------------------------------------------------------------
# Mem0 wrapper
# ---------------------------------------------------------------------------

class Mem0Adapter:
    """Adapter for Mem0 open-source memory system.

    Uses Mem0's Python API to add memories from sessions and search
    for answers to eval queries.

    Mem0 uses OpenAI embeddings by default — requires OPENAI_API_KEY.
    """

    def __init__(
        self,
        user_id: str = "maya",
        results_dir: Optional[Path] = None,
        answer_model: str = "haiku",
        judge_model: str = "gpt-4o-mini",
        ollama_embeddings: bool = False,
        qdrant_dir: Optional[str] = None,
    ):
        self.user_id = user_id
        self.results_dir = results_dir or _DIR.parent / "data" / "results-vm" / "mem0"
        self.answer_model = answer_model
        self.judge_model = judge_model
        self.ollama_embeddings = ollama_embeddings
        self.qdrant_dir = qdrant_dir
        self._mem0 = None
        self._stats = {
            "sessions_injected": 0,
            "messages_added": 0,
            "add_duration_s": 0.0,
            "search_calls": 0,
            "search_duration_s": 0.0,
        }

    @property
    def mem0(self):
        """Lazy-load Mem0 to avoid import errors when not installed."""
        if self._mem0 is None:
            try:
                from mem0 import Memory
                # Use persistent Qdrant storage so data survives across method calls.
                # Qdrant local uses exclusive file locks — only one Memory() instance
                # can access the same path at a time.
                qdrant_path = self.qdrant_dir or str(self.results_dir / ".qdrant")
                collection = "mem0_agentlife"
                if self.ollama_embeddings:
                    collection = "mem0_agentlife_ollama"
                config = {
                    "vector_store": {
                        "provider": "qdrant",
                        "config": {
                            "path": qdrant_path,
                            "collection_name": collection,
                            **({"embedding_model_dims": 4096} if self.ollama_embeddings else {}),
                        },
                    },
                }
                if self.ollama_embeddings:
                    config["embedder"] = {
                        "provider": "ollama",
                        "config": {
                            "model": "qwen3-embedding:8b",
                            "ollama_base_url": "http://localhost:11434",
                        },
                    }
                self._mem0 = Memory.from_config(config)
            except ImportError:
                raise ImportError(
                    "mem0ai not installed. Run: pip install mem0ai\n"
                    "Also requires OPENAI_API_KEY for embeddings."
                )
        return self._mem0

    def inject_sessions(self, reviews: list, per_day: bool = False,
                        per_message_pair: bool = False) -> dict:
        """Inject all sessions into Mem0.

        Processes sessions chronologically, adding each message to Mem0.
        Mem0 handles extraction + embedding internally.

        Args:
            reviews: Chronologically sorted session reviews
            per_day: If True, batch sessions by simulated date (matches
                     Quaid's per-day extraction granularity). If False,
                     inject each session individually.
            per_message_pair: If True, call add() for each user+assistant
                     exchange pair. This is Mem0's intended native mode.
                     Overrides per_day.

        Returns:
            Injection stats dict.
        """
        t0 = time.monotonic()
        total_messages = 0
        total_add_calls = 0

        if per_message_pair:
            # Native Mem0 mode: add() per user+assistant exchange pair
            for review in reviews:
                snum = review.session_num
                label = f"F{abs(snum):03d}" if snum < 0 else f"Session {snum}"
                messages = transcript_to_messages(review)
                if not messages:
                    continue

                # Group into user+assistant pairs
                pairs = []
                i = 0
                while i < len(messages):
                    pair = [messages[i]]
                    if i + 1 < len(messages) and messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
                        pair.append(messages[i + 1])
                        i += 2
                    else:
                        i += 1
                    pairs.append(pair)

                print(f"  {label} ({len(pairs)} pairs)", end="", flush=True)
                session_ok = 0
                for pair in pairs:
                    try:
                        self.mem0.add(
                            pair,
                            user_id=self.user_id,
                            metadata={"session": label},
                        )
                        total_messages += len(pair)
                        total_add_calls += 1
                        session_ok += 1
                    except Exception as e:
                        print(f" [PAIR FAILED: {e}]", end="")
                self._stats["sessions_injected"] += 1
                print(f" [{session_ok}/{len(pairs)} OK]")

        elif per_day:
            # Group sessions by simulated date
            from collections import OrderedDict
            day_batches: OrderedDict[str, list] = OrderedDict()
            for review in reviews:
                snum = review.session_num
                date_str = SESSION_DATES.get(snum, "2026-03-01")
                if date_str not in day_batches:
                    day_batches[date_str] = []
                day_batches[date_str].append(review)

            for date_str, day_reviews in day_batches.items():
                labels = []
                all_messages = []
                for review in day_reviews:
                    snum = review.session_num
                    label = f"F{abs(snum):03d}" if snum < 0 else f"Session {snum}"
                    labels.append(label)
                    messages = transcript_to_messages(review)
                    all_messages.extend(messages)

                if not all_messages:
                    continue

                day_label = f"{date_str} ({', '.join(labels)})"
                print(f"  {day_label} ({len(all_messages)} messages)", end="", flush=True)

                try:
                    self.mem0.add(
                        all_messages,
                        user_id=self.user_id,
                        metadata={"date": date_str, "sessions": ", ".join(labels)},
                    )
                    total_messages += len(all_messages)
                    self._stats["sessions_injected"] += len(day_reviews)
                    print(f" [OK]")
                except Exception as e:
                    print(f" [FAILED: {e}]")
        else:
            for review in reviews:
                snum = review.session_num
                label = f"F{abs(snum):03d}" if snum < 0 else f"Session {snum}"
                messages = transcript_to_messages(review)
                if not messages:
                    continue

                print(f"  {label} ({len(messages)} messages)", end="", flush=True)

                try:
                    self.mem0.add(
                        messages,
                        user_id=self.user_id,
                        metadata={"session": label},
                    )
                    total_messages += len(messages)
                    self._stats["sessions_injected"] += 1
                    print(f" [OK]")
                except Exception as e:
                    print(f" [FAILED: {e}]")

        elapsed = round(time.monotonic() - t0, 1)
        self._stats["messages_added"] = total_messages
        self._stats["add_duration_s"] = elapsed
        self._stats["per_day"] = per_day
        self._stats["per_message_pair"] = per_message_pair
        self._stats["add_calls"] = total_add_calls

        # Save stats
        self.results_dir.mkdir(parents=True, exist_ok=True)
        with open(self.results_dir / "injection_stats.json", "w") as f:
            json.dump(self._stats, f, indent=2)

        return {
            "total_messages": total_messages,
            "total_sessions": self._stats["sessions_injected"],
            "compaction_count": 0,  # Mem0 doesn't have compaction
            "per_day": per_day,
            "day_count": len(day_batches) if per_day else self._stats["sessions_injected"],
            "elapsed_s": elapsed,
        }

    def search(self, query: str, limit: int = 10) -> List[dict]:
        """Search Mem0 memories for a query.

        Returns list of memory dicts with 'text' and 'score' keys.
        """
        t0 = time.monotonic()
        try:
            raw = self.mem0.search(query, user_id=self.user_id, limit=limit)
            self._stats["search_calls"] += 1
            self._stats["search_duration_s"] += time.monotonic() - t0

            # Mem0 v1.0+ returns {"results": [...]}, older returns a list
            result_list = raw.get("results", []) if isinstance(raw, dict) else raw

            # Normalize results format
            memories = []
            for r in result_list:
                if isinstance(r, dict):
                    text = r.get("memory", r.get("text", str(r)))
                    score = r.get("score", 0)
                else:
                    text = str(r)
                    score = 0
                memories.append({"text": text, "score": score})
            return memories

        except Exception as e:
            print(f"    Mem0 search error: {e}")
            return []

    def answer_question(self, question: str) -> Tuple[str, List[str], dict]:
        """Search Mem0 and generate an answer using tool use loop.

        Gives the LLM a mem0_search tool for iterative retrieval,
        matching the multi-turn approach used by Quaid's harness.

        Returns (answer_text, tool_call_names, usage_dict).
        """
        api_key = _get_api_key()
        model_map = {
            "haiku": "claude-haiku-4-5-20251001",
            "sonnet": "claude-sonnet-4-5-20250929",
            "opus": "claude-opus-4-6",
        }
        model_id = model_map.get(self.answer_model, self.answer_model)
        usage_total = {"input_tokens": 0, "output_tokens": 0, "api_calls": 0}

        tools = [{
            "name": "mem0_search",
            "description": (
                "Search memory for information about Maya. "
                "Use specific names and topics for best results."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                },
                "required": ["query"],
            },
        }]

        system_prompt = (
            "You are an AI assistant answering questions about a user named Maya "
            "based on your memory of past conversations. Use the mem0_search tool "
            "to search your memory before answering.\n\n"
            "ANSWER RULES:\n"
            "- ALWAYS search memory first, even if the question seems simple.\n"
            "- Be thorough — include specific names, numbers, dates from memory.\n"
            "- State facts directly. Do not add narrative or caveats.\n"
            "- If you don't have enough information, say "
            "\"I don't have information about that.\""
        )

        messages = [{"role": "user", "content": question}]
        tool_call_names = []
        max_turns = 4

        for turn in range(max_turns):
            payload = {
                "model": model_id,
                "max_tokens": 1024,
                "system": system_prompt,
                "tools": tools,
                "messages": messages,
            }

            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=json.dumps(payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
            )

            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read())
            except Exception as e:
                return f"Error: {e}", tool_call_names, usage_total

            # Track token usage
            usage = data.get("usage", {})
            usage_total["input_tokens"] += usage.get("input_tokens", 0)
            usage_total["output_tokens"] += usage.get("output_tokens", 0)
            usage_total["api_calls"] += 1

            # Process response blocks
            text_parts = []
            tool_uses = []
            for block in data.get("content", []):
                if block["type"] == "text":
                    text_parts.append(block["text"])
                elif block["type"] == "tool_use":
                    tool_uses.append(block)

            # If no tool use, return the text answer
            if not tool_uses:
                return " ".join(text_parts).strip(), tool_call_names, usage_total

            # Execute tool calls and build tool results
            messages.append({"role": "assistant", "content": data["content"]})
            tool_results = []
            for tu in tool_uses:
                tool_call_names.append("mem0_search")
                query = tu["input"].get("query", question)
                memories = self.search(query)
                if memories:
                    result_text = "\n".join(
                        f"{i}. {m['text']}" for i, m in enumerate(memories, 1)
                    )
                else:
                    result_text = "No relevant memories found."
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": result_text,
                })
            messages.append({"role": "user", "content": tool_results})

        # Max turns exhausted — return whatever text we have
        return " ".join(text_parts).strip() if text_parts else "", tool_call_names, usage_total

    def get_stats(self) -> dict:
        """Return accumulated statistics."""
        return self._stats.copy()

    def reset(self):
        """Reset all memories for the user. Use between benchmark runs."""
        try:
            self.mem0.delete_all(user_id=self.user_id)
            print(f"  Mem0 memories cleared for {self.user_id}")
        except Exception as e:
            print(f"  WARNING: Mem0 reset failed: {e}")

        self._stats = {
            "sessions_injected": 0,
            "messages_added": 0,
            "add_duration_s": 0.0,
            "search_calls": 0,
            "search_duration_s": 0.0,
        }


# ---------------------------------------------------------------------------
# Tier 5: Emotional Intelligence
# ---------------------------------------------------------------------------

def _run_tier5_mem0(adapter: Mem0Adapter, results_dir: Path, answer_model: str):
    """Run Tier 5 Emotional Intelligence eval on Mem0."""
    from collections import defaultdict
    from dataset import get_tier5_queries

    queries = get_tier5_queries()
    api_key = _get_api_key()

    print(f"\n{'=' * 60}")
    print(f"TIER 5: EMOTIONAL INTELLIGENCE (Mem0, {answer_model})")
    print(f"{'=' * 60}")
    print(f"  {len(queries)} EI queries")

    results = []
    total_score = 0
    max_possible = len(queries) * 2

    for i, query in enumerate(queries):
        question = query["question"]

        t0 = time.time()
        prediction, tool_calls, usage = adapter.answer_question(question)
        answer_duration = time.time() - t0

        # Judge with Tier 5 rubric (Sonnet)
        ei_score, reasoning = _judge_tier5(query, prediction, api_key)
        total_score += ei_score

        marker = {2: "++", 1: "~", 0: "X"}[ei_score]
        running_pct = total_score / ((i + 1) * 2) * 100
        tools_str = f" tools=[{','.join(tool_calls)}]" if tool_calls else ""
        print(f"  [{i+1}/{len(queries)}] {marker} ({ei_score}/2) {query.get('ei_id', '')} "
              f"{question[:50]}...{tools_str} [{running_pct:.0f}%]")

        results.append({
            "ei_id": query.get("ei_id", f"EI-{i+1:02d}"),
            "ei_category": query.get("ei_category", ""),
            "question": question,
            "prediction": prediction,
            "ei_score": ei_score,
            "reasoning": reasoning,
            "sensitivity_context": query.get("sensitivity_context", ""),
            "rubric": query.get("rubric", {}),
            "tool_calls": tool_calls,
            "answer_duration_s": round(answer_duration, 2),
            "eval_tokens": usage,
        })

    pct = total_score / max_possible * 100 if max_possible > 0 else 0
    print(f"\n{'=' * 60}")
    print(f"Tier 5 Score: {total_score}/{max_possible} ({pct:.1f}%)")
    print(f"{'=' * 60}")

    # Category breakdown
    by_cat = defaultdict(lambda: {"total": 0, "max": 0, "count": 0})
    for r in results:
        cat = r["ei_category"]
        by_cat[cat]["total"] += r["ei_score"]
        by_cat[cat]["max"] += 2
        by_cat[cat]["count"] += 1
    print(f"\n{'Category':<30} {'Score':>8} {'Pct':>6}")
    print(f"{'─' * 50}")
    for cat, s in sorted(by_cat.items()):
        cat_pct = s["total"] / s["max"] * 100 if s["max"] > 0 else 0
        print(f"{cat:<30} {s['total']:>3}/{s['max']:<3} {cat_pct:>5.0f}%")

    # Save
    results_dir.mkdir(parents=True, exist_ok=True)
    tier5_path = results_dir / "tier5_results.json"
    with open(tier5_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {tier5_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Mem0 Adapter for AgentLife")
    parser.add_argument("--test", action="store_true",
                        help="Run a quick integration test")
    parser.add_argument("--reset", action="store_true",
                        help="Clear all Mem0 memories for maya")
    parser.add_argument("--inject", action="store_true",
                        help="Inject sessions and run eval")
    parser.add_argument("--per-day", action="store_true",
                        help="Batch sessions by day (matches Quaid granularity)")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Results directory")
    parser.add_argument("--answer-model", type=str, default="haiku",
                        help="Model for answering queries")
    parser.add_argument("--judge", type=str, default="gpt-4o-mini",
                        choices=["gpt-4o-mini", "haiku"],
                        help="Judge model (default: gpt-4o-mini)")
    parser.add_argument("--ollama-embeddings", action="store_true",
                        help="Use Ollama (qwen3-embedding:8b) instead of OpenAI embeddings")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip injection, eval only (requires existing Qdrant data)")
    parser.add_argument("--qdrant-dir", type=str, default=None,
                        help="Override Qdrant data dir (default: <results-dir>/.qdrant)")
    parser.add_argument("--tier5", action="store_true",
                        help="Run Tier 5 Emotional Intelligence eval (requires existing Qdrant data)")
    parser.add_argument("--tier5-fc", action="store_true",
                        help="Run Tier 5 FC baseline (no memory system, full transcript context)")
    args = parser.parse_args()

    if args.reset:
        results_dir = Path(args.results_dir) if args.results_dir else None
        adapter = Mem0Adapter(
            results_dir=results_dir,
            ollama_embeddings=args.ollama_embeddings,
        ) if results_dir else Mem0Adapter(ollama_embeddings=args.ollama_embeddings)
        adapter.reset()
        return

    if args.inject or args.eval_only:
        from dataset import load_all_reviews, get_all_eval_queries
        results_dir = Path(args.results_dir) if args.results_dir else _DIR.parent / "data" / "results-mem0-perday"
        adapter = Mem0Adapter(results_dir=results_dir, answer_model=args.answer_model,
                              judge_model=args.judge,
                              ollama_embeddings=args.ollama_embeddings,
                              qdrant_dir=args.qdrant_dir)

        # Load sessions
        assets_dir = _DIR.parent.parent.parent / "assets"
        reviews = load_all_reviews(assets_dir)
        queries = get_all_eval_queries(reviews)

        emb_label = "Ollama (qwen3-embedding:8b)" if args.ollama_embeddings else "OpenAI (default)"
        mode = "eval-only" if args.eval_only else ("per-day" if args.per_day else "per-session")
        print(f"Mem0 {mode}")
        print(f"  Embeddings: {emb_label}")
        print(f"  Answer model: {args.answer_model}")
        print(f"  Sessions: {len(reviews)}")
        print(f"  Queries: {len(queries)}")
        print(f"  Results: {results_dir}")
        if args.qdrant_dir:
            print(f"  Qdrant dir: {args.qdrant_dir}")
        print()

        if not args.eval_only:
            # Inject
            stats = adapter.inject_sessions(reviews, per_day=args.per_day)
            print(f"\nInjection: {stats['total_messages']} messages, "
                  f"{stats.get('day_count', stats['total_sessions'])} batches, "
                  f"{stats['elapsed_s']}s")
        else:
            print("Skipping injection (--eval-only), using existing Qdrant data")

        # Eval
        print(f"\nEvaluating {len(queries)} queries...")
        results = []
        total_usage = {"eval_input_tokens": 0, "eval_output_tokens": 0,
                       "eval_api_calls": 0, "judge_api_calls": 0}
        for i, q in enumerate(queries):
            question = q["question"]
            ground_truth = q["ground_truth"]
            print(f"  [{i+1}/{len(queries)}] {question[:60]}...", end="", flush=True)

            prediction, tool_calls, usage = adapter.answer_question(question)
            total_usage["eval_input_tokens"] += usage.get("input_tokens", 0)
            total_usage["eval_output_tokens"] += usage.get("output_tokens", 0)
            total_usage["eval_api_calls"] += usage.get("api_calls", 0)

            # Unified judge (same prompt as all other systems)
            api_key = _get_api_key()
            label, score = _judge(question, ground_truth, prediction, api_key,
                                  judge_model=adapter.judge_model)
            total_usage["judge_api_calls"] += 1

            sym = "O" if score == 1.0 else ("~" if score == 0.5 else "X")
            results.append({
                "question": question, "ground_truth": ground_truth,
                "prediction": prediction, "judge_label": label, "score": score,
                "query_type": q.get("query_type", "unknown"),
                "source_session": q.get("source_session", 0),
                "tool_calls": tool_calls,
                "eval_tokens": usage,
            })
            running = sum(r["score"] for r in results) / len(results) * 100
            tools_str = f" tools=[{','.join(tool_calls)}]" if tool_calls else ""
            print(f" {sym}{tools_str} [{running:.1f}%]")

        # Score
        correct = sum(1 for r in results if r["judge_label"] == "CORRECT")
        partial = sum(1 for r in results if r["judge_label"] == "PARTIAL")
        wrong = sum(1 for r in results if r["judge_label"] == "WRONG")
        accuracy = (correct + 0.5 * partial) / len(results) * 100

        # Token usage summary
        eval_in = total_usage["eval_input_tokens"]
        eval_out = total_usage["eval_output_tokens"]
        eval_calls = total_usage["eval_api_calls"]

        # Cost estimates (Feb 2026 pricing)
        MODEL_COSTS = {
            "haiku": {"input": 0.80, "output": 4.00},    # per 1M tokens
            "sonnet": {"input": 3.00, "output": 15.00},
            "opus": {"input": 15.00, "output": 75.00},
        }
        costs = MODEL_COSTS.get(args.answer_model, MODEL_COSTS["haiku"])
        eval_cost = (eval_in * costs["input"] + eval_out * costs["output"]) / 1_000_000

        print(f"\n{'=' * 50}")
        print(f"Mem0 ({mode}): "
              f"{accuracy:.1f}% ({correct}C/{partial}P/{wrong}W)")
        print(f"Tokens: {eval_in:,} in + {eval_out:,} out = {eval_in + eval_out:,} total")
        print(f"API calls: {eval_calls} eval + {total_usage['judge_api_calls']} judge")
        print(f"Est. eval cost: ${eval_cost:.2f} ({args.answer_model})")
        print(f"{'=' * 50}")

        # Save
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "eval_results.json", "w") as f:
            json.dump(results, f, indent=2)
        with open(results_dir / "scores.json", "w") as f:
            json.dump({
                "accuracy": round(accuracy, 2),
                "correct": correct, "partial": partial, "wrong": wrong,
                "total": len(results), "per_day": args.per_day,
                "judge_model": adapter.judge_model,
                "answer_model": args.answer_model,
                "embeddings": "ollama" if args.ollama_embeddings else "openai",
            }, f, indent=2)
        with open(results_dir / "token_usage.json", "w") as f:
            json.dump({
                "eval": {
                    "input_tokens": eval_in,
                    "output_tokens": eval_out,
                    "api_calls": eval_calls,
                    "model": args.answer_model,
                    "cost_usd": round(eval_cost, 4),
                },
                "judge": {
                    "api_calls": total_usage["judge_api_calls"],
                    "model": adapter.judge_model,
                },
                "total_tokens": eval_in + eval_out,
            }, f, indent=2)
        print(f"Results saved to {results_dir}")

        # Tier 5 if requested (can also run standalone with --tier5 --eval-only)
        if args.tier5:
            _run_tier5_mem0(adapter, results_dir, args.answer_model)
        return

    if args.tier5_fc:
        # FC Tier 5 baseline — no memory system, full transcript context
        from run_production_benchmark import run_tier5_fc_baseline
        results_dir = Path(args.results_dir) if args.results_dir else _DIR.parent / "data" / "results-v10-fc" / "fc_baselines"
        model_map = {"haiku": "claude-haiku-4-5-20251001",
                     "sonnet": "claude-sonnet-4-5-20250929",
                     "opus": "claude-opus-4-6"}
        fc_model = model_map.get(args.answer_model, args.answer_model)
        run_tier5_fc_baseline(
            _get_api_key(),
            answer_model=fc_model,
            results_dir=results_dir,
        )
        return

    if args.tier5:
        # Standalone Tier 5 eval (requires existing Qdrant data)
        results_dir = Path(args.results_dir) if args.results_dir else _DIR.parent / "data" / "results-mem0-perday"
        adapter = Mem0Adapter(
            results_dir=results_dir, answer_model=args.answer_model,
            judge_model=args.judge,
            ollama_embeddings=args.ollama_embeddings,
            qdrant_dir=args.qdrant_dir,
        )
        _run_tier5_mem0(adapter, results_dir, args.answer_model)
        return

    if args.test:
        print("Testing Mem0 adapter...")
        adapter = Mem0Adapter()

        # Add a test memory
        test_messages = [
            {"role": "user", "content": "hey! i'm maya, just moved to austin with my boyfriend david"},
            {"role": "assistant", "content": "Welcome to Austin, Maya! That's exciting. How are you settling in?"},
            {"role": "user", "content": "pretty good! we have a dog named biscuit, a golden retriever"},
            {"role": "assistant", "content": "Biscuit sounds adorable! Golden retrievers are great dogs."},
        ]

        print("  Adding test messages...")
        adapter.mem0.add(test_messages, user_id="test-maya")

        print("  Searching for 'dog'...")
        raw = adapter.mem0.search("What kind of dog does Maya have?", user_id="test-maya")
        result_list = raw.get("results", []) if isinstance(raw, dict) else raw
        for r in result_list[:3]:
            text = r.get("memory", str(r))[:100]
            print(f"    - {text}")

        # Cleanup
        adapter.mem0.delete_all(user_id="test-maya")
        print("  Test complete (memories cleaned up)")


if __name__ == "__main__":
    main()
