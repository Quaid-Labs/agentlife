"""Conftest: stub benchmark-only modules so harness imports work in checkpoint tests."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock


ROOT = Path(__file__).resolve().parents[1]

_dataset_mod = types.ModuleType("dataset")
_dataset_mod.load_all_reviews = MagicMock(return_value=[])
_dataset_mod.get_all_eval_queries = MagicMock(return_value=[])
_dataset_mod.format_transcript_for_extraction = MagicMock(return_value="")
_dataset_mod.SESSION_DATES = {i: "2026-03-01" for i in range(1, 21)}
_dataset_mod.SESSION_TRACKS = {i: 1 for i in range(1, 21)}
_dataset_mod.get_tier5_queries = MagicMock(return_value=[])
sys.modules.setdefault("dataset", _dataset_mod)

_metrics_mod = types.ModuleType("metrics")
_metrics_mod.score_results = MagicMock(return_value={})
_metrics_mod.retrieval_metrics = MagicMock(return_value={})
_metrics_mod.format_report = MagicMock(return_value="")
sys.modules.setdefault("metrics", _metrics_mod)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
