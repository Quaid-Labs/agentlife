#!/usr/bin/env python3
"""Validate parser-safe Japanese review dataset variants (sessions-jp / filler-sessions-jp)."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

os.environ.setdefault("BENCHMARK_DATASET", "jp")
import dataset as ds  # type: ignore

SCAFFOLD_TOKENS = [
    "SECTION 1:",
    "SECTION 2:",
    "SESSION METADATA:",
    "TURN ",
    "SECTION 4:",
    "QUERY ",
    "Ground Truth:",
    "Evidence Session",
    "Supporting Evidence:",
    "Query Type:",
    "Recall Difficulty:",
]

SCAFFOLD_TOKEN_EQUIVALENTS = [
    ("MAYA:", ("MAYA:", "マヤ:")),
    ("AI ASSISTANT:", ("AI ASSISTANT:", "AIアシスタント:")),
]

DIALOGUE_BLOCK_RE = re.compile(
    r"(?ms)^(MAYA:|マヤ:|AI ASSISTANT:|AIアシスタント:)\s*\n(.*?)(?=^\s*(?:AI ASSISTANT:|AIアシスタント:|MAYA:|マヤ:|ANALYSIS:|---|TURN\s+\d+\s*\(|={8,}|END OF FILLER SESSION|$))"
)
QUERY_LINE_RE = re.compile(r'(?m)^QUERY\s+\d+:\s*"([^"]+)"')
GROUND_TRUTH_RE = re.compile(
    r"(?ms)^\s*Ground Truth:\s*(.*?)(?=\n\s*(?:Evidence Session[s]?:|Supporting Evidence:|Query Type:|Recall Difficulty:|QUERY\s+\d+:|QUERY DISTRIBUTION|={8,}|$))"
)
LOWER_ASCII_WORD_RE = re.compile(r"\b[a-z]{4,}\b")


def _count_token_mismatch(src: str, dst: str) -> List[str]:
    mismatches: List[str] = []
    for tok in SCAFFOLD_TOKENS:
        if src.count(tok) != dst.count(tok):
            mismatches.append(f"{tok!r}: src={src.count(tok)} dst={dst.count(tok)}")
    for label, equivalents in SCAFFOLD_TOKEN_EQUIVALENTS:
        src_count = len(re.findall(rf"(?m)^{re.escape(label)}", src))
        dst_count = sum(
            len(re.findall(rf"(?m)^{re.escape(tok)}", dst))
            for tok in equivalents
        )
        if src_count != dst_count:
            mismatches.append(f"{label!r}: src={src_count} dst={dst_count}")
    return mismatches


def _extract_romaji_candidates(text: str) -> List[str]:
    candidates: List[str] = []

    for _label, body in DIALOGUE_BLOCK_RE.findall(text):
        candidates.extend(LOWER_ASCII_WORD_RE.findall(body))

    for q in QUERY_LINE_RE.findall(text):
        candidates.extend(LOWER_ASCII_WORD_RE.findall(q))

    for gt in GROUND_TRUTH_RE.findall(text):
        candidates.extend(LOWER_ASCII_WORD_RE.findall(gt))

    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _load_arc_reviews(path: Path) -> List[ds.SessionReview]:
    nums: List[int] = []
    for fp in sorted(path.glob("session-*-review-*.txt")):
        m = re.match(r"session-(\d+)-review-", fp.name)
        if m:
            nums.append(int(m.group(1)))
    if not nums:
        return []
    return ds.load_all_reviews(path, sessions=sorted(set(nums)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate jp-translated review datasets")
    parser.add_argument("--source-sessions", default="data/sessions")
    parser.add_argument("--translated-sessions", default="data/sessions-jp")
    parser.add_argument("--source-fillers", default="data/filler-sessions")
    parser.add_argument("--translated-fillers", default="data/filler-sessions-jp")
    parser.add_argument("--expect-queries", type=int, default=None)
    parser.add_argument("--expect-tier5", type=int, default=15)
    parser.add_argument("--strict-no-romaji", action="store_true")
    args = parser.parse_args()

    src_sessions = Path(args.source_sessions)
    tr_sessions = Path(args.translated_sessions)
    src_fillers = Path(args.source_fillers)
    tr_fillers = Path(args.translated_fillers)

    errors: List[str] = []
    warnings: List[str] = []

    # Parser checks
    arc_reviews = _load_arc_reviews(tr_sessions)
    filler_reviews = ds.load_filler_reviews(tr_fillers) if tr_fillers.exists() else []
    print(f"parsed arc reviews: {len(arc_reviews)}")
    print(f"parsed filler reviews: {len(filler_reviews)}")

    all_queries = ds.get_all_eval_queries(arc_reviews)
    print(f"query count: {len(all_queries)}")
    if args.expect_queries is not None and len(all_queries) != args.expect_queries:
        errors.append(f"expected queries={args.expect_queries}, got {len(all_queries)}")

    tier5 = ds.get_tier5_queries()
    print(f"tier5 count: {len(tier5)}")
    if len(tier5) != args.expect_tier5:
        errors.append(f"expected tier5={args.expect_tier5}, got {len(tier5)}")

    # Scaffold parity for translated files that exist in both dirs
    def compare_dir(src_dir: Path, dst_dir: Path, pattern: str) -> None:
        if not dst_dir.exists():
            errors.append(f"missing translated dir: {dst_dir}")
            return
        for dst in sorted(dst_dir.glob(pattern)):
            src = src_dir / dst.name
            if not src.exists():
                warnings.append(f"no source counterpart for {dst}")
                continue
            src_txt = src.read_text(encoding="utf-8", errors="ignore")
            dst_txt = dst.read_text(encoding="utf-8", errors="ignore")
            mm = _count_token_mismatch(src_txt, dst_txt)
            if mm:
                errors.append(f"scaffold mismatch in {dst.name}: " + "; ".join(mm))

            romaji = _extract_romaji_candidates(dst_txt)
            if romaji:
                msg = f"romaji-candidate words in {dst.name}: {', '.join(romaji[:12])}"
                if args.strict_no_romaji:
                    errors.append(msg)
                else:
                    warnings.append(msg)

    compare_dir(src_sessions, tr_sessions, "session-*-review-*.txt")
    compare_dir(src_fillers, tr_fillers, "F*-review.txt")

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"- {w}")

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"- {e}")
        raise SystemExit(1)

    print("\nValidation OK")


if __name__ == "__main__":
    main()
