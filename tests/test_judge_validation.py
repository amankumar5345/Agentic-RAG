"""Tests for judge sample export and agreement arithmetic."""

import pytest

from scripts.export_judge_sample import stratified_sample
from scripts.judge_agreement import compute


def make_rows(n, agreement=True):
    rows = []
    judge = {}
    for index in range(n):
        human_correct = index % 2 == 1
        rows.append(
            {
                "case_id": f"q{index}",
                "human_label": "correct" if human_correct else "incorrect",
            }
        )
        judge[f"q{index}"] = human_correct if agreement else not human_correct
    return rows, judge


def test_perfect_agreement_and_kappa():
    rows, judge = make_rows(50, agreement=True)

    result = compute(rows, judge)

    assert result["n"] == 50
    assert result["agreement"] == 1.0
    assert result["kappa"] == 1.0


def test_fewer_than_fifty_labels_fails():
    rows, judge = make_rows(49, agreement=True)

    with pytest.raises(SystemExit):
        compute(rows, judge)


def test_stratified_sample_size():
    cases = [
        {
            "id": f"q{index}",
            "subtype": ["single_hop", "numeric", "hazard"][index % 3],
        }
        for index in range(90)
    ]

    assert len(stratified_sample(cases, 60)) == 60
