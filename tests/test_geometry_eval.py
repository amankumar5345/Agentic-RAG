"""Tests for geometry metrics, replay identity and ue_mode recording."""

import json

import pytest

from bench.draft_cases import _geometry_plans
from litho_eval.geometry_eval import (
    asks_for_size,
    commands_equal,
    latency_block,
    make_spawn_stub,
    replay_distance,
    summarise,
)
from src.core.geometry import replay


def test_replay_of_every_gold_plan_equals_itself():
    for plan in _geometry_plans():
        if plan["expect_clarification"]:
            continue
        result = replay(plan["commands"])
        again = replay(plan["commands"])
        assert result == again
        assert replay_distance(plan["commands"], plan["commands"]) == 0.0


def test_exact_match_normalises_floats():
    gold = [{"shape": "line", "size": 10.0, "angle": 0.0}]
    prediction = [{"shape": "line", "size": 10.0000004, "angle": 360.0}]

    assert commands_equal(prediction, gold)
    assert not commands_equal([{"shape": "line", "size": 10.1, "angle": 0.0}], gold)


def test_geometric_match_uses_tolerance():
    gold = [
        {"shape": "line", "size": 100.0, "angle": 0.0},
        {"shape": "line", "size": 100.0, "angle": 90.0},
    ]
    prediction = [
        {"shape": "line", "size": 100.0, "angle": 0.0},
        {"shape": "line", "size": 100.0, "angle": 90.0},
    ]

    assert replay_distance(prediction, gold) == 0.0

    shifted = [
        {"shape": "line", "size": 100.0, "angle": 0.0},
        {"shape": "line", "size": 100.5, "angle": 90.0},
    ]
    assert replay_distance(shifted, gold) == pytest.approx(0.5)
    assert replay_distance(shifted, gold) > 0.1


def test_asks_for_size():
    assert asks_for_size("What size should the square be?")
    assert asks_for_size("Please provide the length.")
    assert not asks_for_size("Here is your square.")


def test_latency_block_records_ue_mode():
    block = latency_block({"total_s": [0.1, 0.2]}, "fake")

    assert block["ue_mode"] == "fake"
    assert block["total"]["p50"] == pytest.approx(0.15)


def test_spawn_stub_captures_without_writing(tmp_path):
    captured = []
    stub = make_spawn_stub(captured)

    result = json.loads(stub.invoke({"commands": [{"shape": "line", "size": 10, "angle": 0}]}))

    assert result["status"] == "ok"
    assert captured == [[{"shape": "line", "size": 10, "angle": 0}]]
    assert list(tmp_path.iterdir()) == []


def test_summarise_drawable_and_clarification():
    drawable = {
        "expect_clarification": False,
        "schema_valid": True,
        "exact_match": True,
        "geometric_match": True,
        "clarification_correct": False,
    }
    clarification = {
        "expect_clarification": True,
        "schema_valid": False,
        "exact_match": False,
        "geometric_match": False,
        "clarification_correct": True,
    }

    summary = summarise([drawable, clarification], [1.0, 0.0])

    assert summary["schema_valid"]["rate"] == 1.0
    assert summary["clarification_correctness"]["rate"] == 1.0
    assert summary["over_trigger"]["rate"] == 0.5
