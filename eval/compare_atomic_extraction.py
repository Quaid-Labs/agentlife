#!/usr/bin/env python3
"""Compare benchmark extraction prompt variants on the AL-S corpus.

This is an offline experiment harness. It mirrors the chunked AL-S extraction
path used by the benchmark, but it does not store anything into a workspace DB.
It only compares model output quality and coverage.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extract_compact import build_extraction_prompt, call_anthropic, parse_extraction_response
from run_production_benchmark import _build_session_blocks, _load_reviews_with_dataset_gate, _split_session_blocks_on_gap

CANONICAL_DOMAINS = ["personal", "technical", "projects", "research"]
MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
}

DERIVED_OUTPUT_SPEC = """
OPTIONAL DERIVED FACTS:
- You MAY include a top-level `derived_facts` array in addition to `facts`.
- `facts` must remain ONLY explicit or directly confirmed atomic facts.
- Put something in `derived_facts` only if it is a near-trivial entailment from explicit facts.
- Never put emotions, disputed timeline state, or speculative interpretation in `derived_facts`.
- Never move explicit facts out of `facts` into `derived_facts`.

Derived fact shape:
{
  "text": "the directly entailed fact",
  "basis": ["supporting explicit fact 1", "supporting explicit fact 2"],
  "derivation_confidence": "low|medium"
}
"""

ATOMIC_APPENDIX = """
ATOMIC EXTRACTION MODE:
- Optimize for retrieval handles, not elegance.
- Decompose every explicit claim into the smallest self-contained fact that can stand alone.
- If one sentence contains multiple details, emit multiple facts.
- Prefer three short facts over one polished summary.
- Preserve "small" details when explicit: names, dates, titles, neighborhoods, API names, version numbers, restaurants, pets, coworker names, health metrics, and one-off behaviors.
- Repeat entities across facts when needed for clarity. Do not merge for brevity.
- Do not compress timeline changes into a single blended statement.
- For relationship-heavy lines, separately capture who each person is, how they relate, and any named locations or employers.

ATOMIC QUALITY CHECK:
- Before finalizing, split any fact that contains two independent searchable claims.
- Keep reason clauses separate when possible unless the point is specifically a decision-with-reason fact.
"""

SENTINELS = {
    "biscuit": ["biscuit"],
    "golden_retriever": ["golden retriever"],
    "houston": ["houston"],
    "stripe": ["stripe"],
    "senior_pm": ["senior pm", "senior product manager"],
    "edamam": ["edamam"],
    "fooddata": ["fooddata central", "food data central"],
    "sap_s": ["sap's", "saps"],
    "south_congress": ["south congress"],
    "a1c": ["a1c"],
    "priya": ["priya"],
}

MULTI_CLAUSE_MARKERS = [
    " and ",
    ";",
    " while ",
    " but ",
    ", and ",
]


@dataclass(frozen=True)
class Variant:
    name: str
    atomic_mode: bool
    allow_derived: bool = False
    strict_entities: bool = False


def _load_credential() -> str:
    token = (
        os.environ.get("BENCHMARK_ANTHROPIC_OAUTH_TOKEN")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        or ""
    ).strip()
    if not token:
        raise SystemExit(
            "Missing Anthropic credential. Set BENCHMARK_ANTHROPIC_OAUTH_TOKEN or ANTHROPIC_API_KEY."
        )
    return token


def _variant_prompt(variant: Variant) -> str:
    prompt = build_extraction_prompt("Maya", "Assistant", allowed_domains=CANONICAL_DOMAINS)
    pieces = [prompt]
    if variant.atomic_mode:
        pieces.append(ATOMIC_APPENDIX.strip())
    if variant.strict_entities:
        pieces.append(
            """
STRICT ENTITY AND LABEL RULES:
- Never use initials, shorthand, or partial names when the full entity name is known. Use "David", not "D".
- Never use slash-combined labels like "spouse/partner" or "partner/husband".
- Prefer one canonical relationship label per fact.
- Do not emit a fact if the subject or object would be ambiguous without context.
- Keep project names, API names, and version labels exact when explicitly stated.
""".strip()
        )
    if variant.name == "atomic_canonical":
        pieces.append(
            """
CANONICAL NAME NORMALIZATION CHECK:
- Before finalizing, scan every fact and replace initials or role-only references with the canonical full name whenever the transcript gives it.
- If both a role label and a proper name are available, prefer the proper name in the fact text.
- Do not emit any fact that starts with a single-letter subject.
- Do not emit duplicate relationship facts with alternate wording. Pick the clearest single phrasing.
- If a fact cannot be rewritten into a canonical, self-contained form, drop it.
""".strip()
        )
    if variant.allow_derived:
        pieces.append(DERIVED_OUTPUT_SPEC.strip())
    return "\n\n".join(pieces) + "\n"


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _word_count(text: str) -> int:
    return len(str(text or "").split())


def _multi_clause(text: str) -> bool:
    lower = str(text or "").lower()
    return any(marker in lower for marker in MULTI_CLAUSE_MARKERS)


def _sentinel_hits(facts: Iterable[dict]) -> dict[str, int]:
    joined = "\n".join(str(f.get("text", "")) for f in facts).lower()
    out = {}
    for key, terms in SENTINELS.items():
        out[key] = int(any(term in joined for term in terms))
    return out


def _summary_for(result: dict) -> dict:
    facts = result.get("facts", [])
    derived = result.get("derived_facts", [])
    texts = [str(f.get("text", "")).strip() for f in facts if str(f.get("text", "")).strip()]
    normalized = [_normalize_text(t) for t in texts]
    words = [_word_count(t) for t in texts]
    edges = sum(len(f.get("edges", []) or []) for f in facts)
    categories = Counter((f.get("category") or "unknown").lower() for f in facts)
    confidences = Counter((f.get("extraction_confidence") or "unknown").lower() for f in facts)
    unique_ratio = (len(set(normalized)) / len(normalized)) if normalized else 0.0
    multi_clause_ratio = (
        sum(1 for t in texts if _multi_clause(t)) / len(texts)
        if texts else 0.0
    )
    long_fact_ratio = (sum(1 for w in words if w >= 18) / len(words)) if words else 0.0
    return {
        "facts_total": len(facts),
        "derived_total": len(derived) if isinstance(derived, list) else 0,
        "edges_total": edges,
        "categories": dict(sorted(categories.items())),
        "confidences": dict(sorted(confidences.items())),
        "avg_words": round(statistics.mean(words), 2) if words else 0.0,
        "median_words": statistics.median(words) if words else 0.0,
        "multi_clause_ratio": round(multi_clause_ratio, 4),
        "long_fact_ratio": round(long_fact_ratio, 4),
        "unique_ratio": round(unique_ratio, 4),
        "sentinels": _sentinel_hits(facts),
    }


def _build_jobs(max_sessions: int, gap_seconds: int) -> list[dict]:
    _assets_dir, _arc_reviews, reviews, _dataset_version, _expected_queries = _load_reviews_with_dataset_gate(max_sessions)
    blocks = _build_session_blocks(reviews)
    chunks = _split_session_blocks_on_gap(blocks, gap_seconds)
    jobs = []
    for idx, chunk in enumerate(chunks):
        jobs.append(
            {
                "chunk_idx": idx,
                "sessions": [item["session_num"] for item in chunk],
                "user_message": (
                    "Extract memorable facts from these conversation sessions with Maya.\n\n"
                    + "\n\n".join(item["block"] for item in chunk)
                ),
            }
        )
    return jobs


def _run_variant(
    *,
    model_name: str,
    model_id: str,
    variant: Variant,
    jobs: list[dict],
    credential: str,
    out_dir: Path,
) -> dict:
    prompt = _variant_prompt(variant)
    combo_dir = out_dir / f"{model_name}-{variant.name}"
    combo_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = combo_dir / "chunks"
    raw_dir.mkdir(exist_ok=True)
    print(f"\n== {model_name} / {variant.name} ==")
    aggregate_facts = []
    aggregate_derived = []
    usage_totals = Counter()
    chunk_summaries = []
    t0 = time.time()
    for job in jobs:
        chunk_label = f"chunk-{job['chunk_idx']:03d}"
        print(f"  {chunk_label} sessions={job['sessions']}")
        t_chunk = time.time()
        raw, usage = call_anthropic(prompt, job["user_message"], model_id, credential, max_tokens=32768)
        parsed = parse_extraction_response(raw)
        elapsed = time.time() - t_chunk
        usage_totals.update({
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
        })
        usage_totals["api_calls"] += 1
        usage_totals["elapsed_ms"] += int(elapsed * 1000)
        facts = parsed.get("facts") or []
        derived = parsed.get("derived_facts") or []
        aggregate_facts.extend(facts)
        if isinstance(derived, list):
            aggregate_derived.extend(derived)
        payload = {
            "sessions": job["sessions"],
            "elapsed_s": round(elapsed, 3),
            "usage": usage,
            "parsed": parsed,
            "raw": raw,
        }
        (raw_dir / f"{chunk_label}.json").write_text(json.dumps(payload, indent=2))
        chunk_summaries.append(
            {
                "chunk_idx": job["chunk_idx"],
                "sessions": job["sessions"],
                "facts": len(facts),
                "derived_facts": len(derived) if isinstance(derived, list) else 0,
                "elapsed_s": round(elapsed, 3),
            }
        )
    total_elapsed = time.time() - t0
    aggregate = {
        "model_name": model_name,
        "model_id": model_id,
        "variant": variant.name,
        "atomic_mode": variant.atomic_mode,
        "prompt": prompt,
        "facts": aggregate_facts,
        "derived_facts": aggregate_derived,
        "usage": dict(usage_totals),
        "elapsed_s": round(total_elapsed, 3),
        "chunk_summaries": chunk_summaries,
        "summary": _summary_for({"facts": aggregate_facts, "derived_facts": aggregate_derived}),
    }
    (combo_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2))
    return aggregate


def _sample_delta_texts(base: list[dict], variant: list[dict], *, limit: int = 12) -> list[str]:
    base_norm = {_normalize_text(f.get("text", "")) for f in base if f.get("text")}
    extra = []
    for fact in variant:
        text = str(fact.get("text", "")).strip()
        if not text:
            continue
        if _normalize_text(text) in base_norm:
            continue
        extra.append(text)
    extra.sort(key=lambda t: (_word_count(t), t.lower()))
    return extra[:limit]


def _print_pair_report(model_name: str, baseline: dict, atomic: dict) -> None:
    b = baseline["summary"]
    a = atomic["summary"]
    print(f"\n## {model_name} delta")
    print(
        f"facts {b['facts_total']} -> {a['facts_total']} | "
        f"derived {b['derived_total']} -> {a['derived_total']} | "
        f"edges {b['edges_total']} -> {a['edges_total']}"
    )
    print(
        f"avg_words {b['avg_words']} -> {a['avg_words']} | "
        f"multi_clause {b['multi_clause_ratio']:.2%} -> {a['multi_clause_ratio']:.2%} | "
        f"unique_ratio {b['unique_ratio']:.2%} -> {a['unique_ratio']:.2%}"
    )
    print(f"sentinels baseline={b['sentinels']}")
    print(f"sentinels atomic={a['sentinels']}")
    print("sample new atomic facts:")
    for text in _sample_delta_texts(baseline["facts"], atomic["facts"]):
        print(f"  - {text}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare AL-S extraction prompt variants.")
    parser.add_argument("--max-sessions", type=int, default=20)
    parser.add_argument("--gap-seconds", type=int, default=3600)
    parser.add_argument(
        "--models",
        default="haiku,sonnet",
        help="Comma-separated model keys from {haiku,sonnet}",
    )
    parser.add_argument(
        "--variants",
        default="baseline,atomic",
        help="Comma-separated variant names from {baseline,atomic,atomic_strict}",
    )
    parser.add_argument(
        "--out-dir",
        default=str((ROOT.parent / "tmp" / f"extraction-compare-{datetime.now().strftime('%Y%m%d-%H%M%S')}")),
    )
    args = parser.parse_args()

    credential = _load_credential()
    jobs = _build_jobs(args.max_sessions, args.gap_seconds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")
    print(f"Chunks: {len(jobs)}")

    available_variants = {
        "baseline": Variant("baseline", atomic_mode=False),
        "atomic": Variant("atomic", atomic_mode=True, allow_derived=True),
        "atomic_strict": Variant("atomic_strict", atomic_mode=True, allow_derived=False, strict_entities=True),
        "atomic_canonical": Variant("atomic_canonical", atomic_mode=True, allow_derived=False, strict_entities=True),
    }
    selected_model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    selected_variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if not selected_model_names:
        raise SystemExit("No models selected")
    if not selected_variants:
        raise SystemExit("No variants selected")
    for model_name in selected_model_names:
        if model_name not in MODELS:
            raise SystemExit(f"Unknown model key: {model_name}")
    for variant_name in selected_variants:
        if variant_name not in available_variants:
            raise SystemExit(f"Unknown variant: {variant_name}")

    results = {}
    for model_name in selected_model_names:
        model_id = MODELS[model_name]
        for variant_name in selected_variants:
            variant = available_variants[variant_name]
            aggregate = _run_variant(
                model_name=model_name,
                model_id=model_id,
                variant=variant,
                jobs=jobs,
                credential=credential,
                out_dir=out_dir,
            )
            results[(model_name, variant.name)] = aggregate

    summary = {}
    for (model_name, variant_name), aggregate in results.items():
        summary[f"{model_name}-{variant_name}"] = {
            "model_id": aggregate["model_id"],
            "summary": aggregate["summary"],
            "usage": aggregate["usage"],
            "elapsed_s": aggregate["elapsed_s"],
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if ("haiku", "baseline") in results and ("haiku", "atomic") in results:
        _print_pair_report("haiku", results[("haiku", "baseline")], results[("haiku", "atomic")])
    if ("sonnet", "baseline") in results and ("sonnet", "atomic") in results:
        _print_pair_report("sonnet", results[("sonnet", "baseline")], results[("sonnet", "atomic")])
    print(f"\nWrote: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
