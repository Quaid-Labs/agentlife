#!/usr/bin/env python3
"""AgentLife Benchmark — Evaluation pipeline.

Recall memories + generate answers + judge predictions.
Supports both memory-based and full-context baselines.
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from string import Template
from typing import Dict, List, Optional, Tuple

_DIR = Path(__file__).resolve().parent
_WORKSPACE = Path(os.environ.get("CLAWDBOT_WORKSPACE", Path.home() / "clawd"))
_QUAID_DIR = _WORKSPACE / "plugins" / "quaid"
_RUNNER_DIR = _WORKSPACE / "memory-stress-test" / "runner"

if str(_QUAID_DIR) not in sys.path:
    sys.path.insert(0, str(_QUAID_DIR))
if str(_RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(_RUNNER_DIR))

from claude_backend import call_claude
from dataset import SessionReview, format_transcript_for_fullcontext

# ---------------------------------------------------------------------------
# Token tracking
# ---------------------------------------------------------------------------

_eval_counts = {
    "recall_calls": 0,
    "answer_calls": 0,
    "judge_calls": 0,
    "answer_input_tokens_est": 0,
    "answer_output_tokens_est": 0,
    "context_tokens_total": 0,
    "total_duration_s": 0.0,
}


def get_eval_stats() -> dict:
    return _eval_counts.copy()


def reset_eval_stats():
    global _eval_counts
    _eval_counts = {k: 0 for k in _eval_counts}
    _eval_counts["total_duration_s"] = 0.0


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ANSWER_PROMPT = Template("""\
You are an AI assistant answering questions about a user named Maya based on your memory of past conversations.

Here is what you remember:

$context

Based on this context, answer the following question concisely and accurately.
If you don't have enough information to answer, say "I don't have information about that."

Question: $question

Answer:""")

FC_ANSWER_PROMPT = Template("""\
You are an AI assistant answering questions about a user named Maya. Below are transcripts of your past conversations with Maya.

$transcripts

Based on these conversations, answer the following question concisely and accurately.
If the conversations don't contain enough information, say "I don't have information about that."

Question: $question

Answer:""")

# GPT-4o-mini judge prompt (matches Mem0/LoCoMo methodology)
JUDGE_PROMPT = Template("""\
You are evaluating the accuracy of a memory system's response.

Question: $question
Correct Answer: $ground_truth
Model Response: $prediction

Score the model's response:
- CORRECT: The response contains the correct information and answers the question accurately
- PARTIAL: The response contains some correct information but is incomplete or has extras
- WRONG: The response is incorrect, missing key information, or hallucinated

Respond with exactly one word: CORRECT, PARTIAL, or WRONG.""")


# ---------------------------------------------------------------------------
# DB switching (reuse from ingest)
# ---------------------------------------------------------------------------

def _switch_to_db(db_path: Path):
    """Switch Quaid to the specified DB.

    CRITICAL: Must pass db_path explicitly to MemoryGraph() constructor.
    DB_PATH default param is captured at import time.
    """
    os.environ["MEMORY_DB_PATH"] = str(db_path)
    import memory_graph
    memory_graph._graph = memory_graph.MemoryGraph(db_path=db_path)


# ---------------------------------------------------------------------------
# Memory recall
# ---------------------------------------------------------------------------

def recall_memories(
    query: str,
    owner_id: str = "maya",
    top_k: int = 10,
    recall_kwargs: Optional[dict] = None,
) -> Tuple[List[dict], float]:
    """Recall memories from the DB for a given query.

    Returns (memories_list, latency_ms).
    """
    from memory_graph import recall

    kwargs = {
        "query": query,
        "limit": top_k,
        "owner_id": owner_id,
    }
    if recall_kwargs:
        kwargs.update(recall_kwargs)

    t0 = time.monotonic()
    try:
        memories = recall(**kwargs)
    except Exception as e:
        print(f"    WARNING: Recall failed: {e}")
        memories = []
    latency_ms = (time.monotonic() - t0) * 1000

    # Filter out low-value results
    filtered = []
    for m in memories:
        text = m.get("text", "") if isinstance(m, dict) else str(m)
        if len(text) >= 5:
            filtered.append(m)

    _eval_counts["recall_calls"] += 1
    return filtered, latency_ms


def search_project_docs(
    query: str,
    results_dir: Path,
    top_k: int = 5,
) -> List[str]:
    """Search project docs via RAG."""
    try:
        from docs_rag import DocsRAG
        rag = DocsRAG()
        results = rag.search_docs(query, limit=top_k)
        return [r.get("text", "") for r in results if r.get("text")]
    except Exception:
        # Fallback: read PROJECT.md directly
        project_md = results_dir / "projects" / "recipe-app" / "PROJECT.md"
        if project_md.exists():
            return [project_md.read_text()[:2000]]
        return []


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def build_context(
    memories: List[dict],
    project_docs: List[str] = None,
) -> str:
    """Build the context string from memories + project docs."""
    parts = []

    if memories:
        parts.append("## Retrieved Memories")
        for i, m in enumerate(memories, 1):
            text = m.get("text", "") if isinstance(m, dict) else str(m)
            conf = m.get("confidence", 0) if isinstance(m, dict) else 0
            parts.append(f"{i}. {text} (confidence: {conf:.1f})")

    if project_docs:
        parts.append("\n## Project Documentation")
        for doc in project_docs:
            parts.append(doc[:1500])  # Truncate long docs

    return "\n".join(parts) if parts else "No memories found."


def generate_answer(
    question: str,
    context: str,
    answer_model: str = "haiku",
) -> Tuple[str, float]:
    """Generate an answer using the LLM.

    Returns (answer_text, duration_seconds).
    """
    prompt = ANSWER_PROMPT.safe_substitute(
        context=context,
        question=question,
    )

    response, duration = call_claude(
        prompt=prompt,
        model=answer_model,
        timeout=60,
    )

    _eval_counts["answer_calls"] += 1
    _eval_counts["answer_input_tokens_est"] += len(prompt) // 4
    _eval_counts["answer_output_tokens_est"] += len(response or "") // 4
    _eval_counts["context_tokens_total"] += len(context) // 4

    return (response or "").strip(), duration


def generate_fc_answer(
    question: str,
    all_transcripts: str,
    answer_model: str = "haiku",
) -> Tuple[str, float]:
    """Generate an answer with full conversation context (FC baseline).

    Returns (answer_text, duration_seconds).
    """
    prompt = FC_ANSWER_PROMPT.safe_substitute(
        transcripts=all_transcripts,
        question=question,
    )

    response, duration = call_claude(
        prompt=prompt,
        model=answer_model,
        timeout=120,
    )

    _eval_counts["answer_calls"] += 1
    _eval_counts["answer_input_tokens_est"] += len(prompt) // 4
    _eval_counts["answer_output_tokens_est"] += len(response or "") // 4

    return (response or "").strip(), duration


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------

def judge_prediction(
    question: str,
    ground_truth: str,
    prediction: str,
    judge_model: str = "haiku",  # Use haiku for speed; GPT-4o-mini for publication
) -> Tuple[str, float, float]:
    """Judge a prediction against ground truth.

    Returns (label, score, duration).
    label: CORRECT, PARTIAL, WRONG, or ERROR
    score: 1.0, 0.5, 0.0
    """
    if not prediction or prediction.strip().lower() in ("", "i don't know", "n/a"):
        return "WRONG", 0.0, 0.0

    prompt = JUDGE_PROMPT.safe_substitute(
        question=question,
        ground_truth=ground_truth,
        prediction=prediction,
    )

    response, duration = call_claude(
        prompt=prompt,
        model=judge_model,
        timeout=30,
    )

    _eval_counts["judge_calls"] += 1

    if not response:
        return "ERROR", 0.0, duration

    label = response.strip().upper()
    if "CORRECT" in label:
        return "CORRECT", 1.0, duration
    elif "PARTIAL" in label:
        return "PARTIAL", 0.5, duration
    elif "WRONG" in label:
        return "WRONG", 0.0, duration
    else:
        return "ERROR", 0.0, duration


# ---------------------------------------------------------------------------
# Evaluation orchestration
# ---------------------------------------------------------------------------

def evaluate_single(
    query: dict,
    results_dir: Path,
    answer_model: str = "haiku",
    judge_model: str = "haiku",
    owner_id: str = "maya",
    top_k: int = 10,
    recall_kwargs: Optional[dict] = None,
    include_project_docs: bool = True,
) -> dict:
    """Evaluate a single query.

    Returns a result dict with question, ground_truth, prediction, score, etc.
    """
    question = query["question"]
    ground_truth = query["ground_truth"]

    # 1. Recall memories
    memories, recall_ms = recall_memories(question, owner_id, top_k, recall_kwargs)

    # 2. Search project docs (for project-related queries)
    project_docs = []
    if include_project_docs:
        project_docs = search_project_docs(question, results_dir)

    # 3. Build context
    context = build_context(memories, project_docs)

    # 4. Generate answer
    prediction, answer_duration = generate_answer(question, context, answer_model)

    # 5. Judge
    label, score, judge_duration = judge_prediction(
        question, ground_truth, prediction, judge_model
    )

    return {
        "question": question,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "judge_label": label,
        "score": score,
        "query_type": query.get("query_type", "unknown"),
        "recall_difficulty": query.get("recall_difficulty", "unknown"),
        "source_session": query.get("source_session", 0),
        "evidence_sessions": query.get("evidence_sessions", []),
        "num_memories": len(memories),
        "num_project_docs": len(project_docs),
        "context_tokens_est": len(context) // 4,
        "recall_latency_ms": round(recall_ms, 1),
        "answer_duration_s": round(answer_duration, 2),
        "judge_duration_s": round(judge_duration, 2),
        "retrieved_memories": [
            {"text": m.get("text", "")[:200], "confidence": m.get("confidence", 0)}
            for m in memories[:5]
        ],
    }


def evaluate_all(
    queries: List[dict],
    results_dir: Path,
    db_path: Path,
    answer_model: str = "haiku",
    judge_model: str = "haiku",
    owner_id: str = "maya",
    top_k: int = 10,
    recall_kwargs: Optional[dict] = None,
) -> List[dict]:
    """Evaluate all queries against the ingested DB.

    Returns list of result dicts.
    """
    reset_eval_stats()
    _switch_to_db(db_path)

    results = []
    for i, query in enumerate(queries):
        print(f"  [{i+1}/{len(queries)}] {query['question'][:60]}...")
        try:
            result = evaluate_single(
                query, results_dir, answer_model, judge_model,
                owner_id, top_k, recall_kwargs,
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            result = {
                "question": query["question"],
                "ground_truth": query["ground_truth"],
                "prediction": "",
                "judge_label": "ERROR",
                "score": 0.0,
                "query_type": query.get("query_type", "unknown"),
                "error": str(e),
            }
        results.append(result)

    return results


def evaluate_fullcontext(
    queries: List[dict],
    reviews: List[SessionReview],
    answer_model: str = "haiku",
    judge_model: str = "haiku",
) -> List[dict]:
    """Full-context baseline: dump all transcripts into prompt.

    Returns list of result dicts.
    """
    reset_eval_stats()
    all_transcripts = format_transcript_for_fullcontext(reviews)

    results = []
    for i, query in enumerate(queries):
        question = query["question"]
        ground_truth = query["ground_truth"]
        print(f"  FC [{i+1}/{len(queries)}] {question[:60]}...")

        try:
            prediction, duration = generate_fc_answer(question, all_transcripts, answer_model)
            label, score, judge_duration = judge_prediction(
                question, ground_truth, prediction, judge_model
            )
        except Exception as e:
            print(f"    FC ERROR: {e}")
            prediction, label, score = "", "ERROR", 0.0
            duration = 0.0

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "judge_label": label,
            "score": score,
            "query_type": query.get("query_type", "unknown"),
            "recall_difficulty": query.get("recall_difficulty", "unknown"),
            "source_session": query.get("source_session", 0),
            "context_tokens_est": len(all_transcripts) // 4,
            "answer_duration_s": round(duration, 2),
            "mode": "full_context",
        })

    return results
