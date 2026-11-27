"""Unit tests for UI loaders and chart builders."""

from __future__ import annotations

import pandas as pd

from src.ui.components.charts import (
    bar_with_error,
    confusion_heatmap,
    csv_bytes,
    latency_timeline,
    scatter_cost_latency,
)
from src.ui.components.mock import data_status, with_status
from src.ui.data import (
    derived_difficulty,
    difficulty_view,
    is_mock,
    load_result,
    results_manifest,
    status_label,
)
from src.ui.state import resolve_mode


def frame():
    return pd.DataFrame(
        [
            {"arm": "A0", "correctness": 0.8, "lo": 0.7, "hi": 0.9, "cost": 0.0, "p95": 0.1},
            {"arm": "A1", "correctness": 0.9, "lo": 0.8, "hi": 1.0, "cost": 0.01, "p95": 0.2},
        ]
    )



def test_resolve_mode_defaults_to_demo_without_keys(monkeypatch):
    monkeypatch.delenv("LITHO_MODE", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    mode, reason = resolve_mode()

    assert mode == "demo"
    assert reason == ""


def test_resolve_mode_live_with_keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "y")
    monkeypatch.delenv("LITHO_MODE", raising=False)

    assert resolve_mode()[0] == "live"


def test_missing_result_is_not_run():
    payload = load_result("judge_validation")

    assert status_label(payload) == "NOT YET RUN"


def test_mock_result_is_flagged():
    assert is_mock({"mock": True})
    assert data_status({"mock": True}) == "MOCK"
    assert data_status({"status": "blocked"}) == "NOT YET RUN"
    assert data_status({"mock": False}) == "MEASURED"


def test_with_status_adds_a_visible_data_column():
    labelled = with_status(pd.DataFrame([{"arm": "A1"}]), {"mock": True})

    assert list(labelled["data"]) == ["MOCK"]


def test_derived_difficulty():
    assert derived_difficulty({"type": "qa", "subtype": "multi_hop"}) == "hard"
    assert derived_difficulty({"type": "qa", "subtype": "single_hop"}) == "medium"
    assert difficulty_view({"type": "qa", "subtype": "multi_hop"})[1] == "derived (heuristic)"
    assert difficulty_view({"type": "qa", "subtype": "hazard"}) == ("not set", "not set")


def test_stored_difficulty_wins_over_heuristic():
    assert difficulty_view({"difficulty": "easy", "subtype": "multi_hop"}) == (
        "easy",
        "stored",
    )


def test_manifest_lists_results():
    names = {row["result"] for row in results_manifest()}

    assert "comparison" in names
    assert "judge_validation" in names


def test_chart_builders():
    data = frame()
    bar = bar_with_error(data, "arm", "correctness", "lo", "hi", "Correctness", "score")
    scatter = scatter_cost_latency(data)
    heat = confusion_heatmap({"jailbreak": {"tp": 1, "fp": 0, "fn": 1, "tn": 0}}, "Confusion")
    timeline = latency_timeline({"guardrail_s": 0.1, "total_s": 0.2})

    assert len(bar.data) == 1
    assert len(scatter.data) == 1
    assert len(heat.data) == 1
    assert len(timeline.data) == 1
    assert csv_bytes(data).startswith(b"arm,")
