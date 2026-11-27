"""Mock results must never reach generated README blocks."""

from __future__ import annotations

import json

import scripts.build_readme as build_readme


def _write(results, name: str, payload: dict) -> None:
    (results / name).write_text(json.dumps(payload), encoding="utf-8")


def test_mock_arms_never_render(monkeypatch, tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    _write(
        results,
        "comparison.json",
        {
            "mock": True,
            "baseline": "A1",
            "arms": {
                "A2": {
                    "metrics": {
                        "answer_correctness": {
                            "delta": 0.987654,
                            "ci95": [0.1, 0.2],
                            "n": 5,
                        }
                    }
                }
            },
        },
    )
    _write(results, "A1.json", {"mock": True})
    monkeypatch.setattr(build_readme, "RESULTS", results)

    body = build_readme.arms_block()

    assert body == build_readme.NOT_RUN
    assert "0.987654" not in body


def test_missing_arms_block_is_not_run(monkeypatch, tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr(build_readme, "RESULTS", results)

    assert build_readme.arms_block() == build_readme.NOT_RUN


def test_measured_arms_render_when_no_mock_file_exists(monkeypatch, tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    _write(
        results,
        "comparison.json",
        {
            "mock": False,
            "baseline": "A1",
            "arms": {
                "A2": {
                    "metrics": {
                        "answer_correctness": {
                            "delta": 0.25,
                            "ci95": [0.1, 0.4],
                            "n": 20,
                        }
                    }
                }
            },
        },
    )
    _write(results, "A1.json", {"mock": False})
    _write(results, "A2.json", {"mock": False})
    monkeypatch.setattr(build_readme, "RESULTS", results)

    body = build_readme.arms_block()

    assert "0.25" in body
    assert build_readme.NOT_RUN not in body


def test_mock_guardrails_never_render(monkeypatch, tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    _write(
        results,
        "guardrails.json",
        {
            "mock": True,
            "status": "ok",
            "systems": {"G1": {"asr": 0.123456, "false_refusal": 0.02}},
        },
    )
    monkeypatch.setattr(build_readme, "RESULTS", results)

    body = build_readme.guardrails_block()

    assert body == build_readme.NOT_RUN
    assert "0.123456" not in body


def test_blocked_a8_is_marked_not_run_without_a_delta(monkeypatch, tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    _write(
        results,
        "comparison.json",
        {
            "mock": False,
            "baseline": "A1",
            "arms": {
                "A2": {
                    "metrics": {
                        "answer_correctness": {
                            "delta": 0.25,
                            "ci95": [0.1, 0.4],
                            "n": 20,
                        }
                    }
                },
                "A8": {
                    "metrics": {
                        "answer_correctness": {
                            "delta": 0.9,
                            "ci95": [0.8, 1.0],
                            "n": 20,
                        }
                    }
                },
            },
        },
    )
    _write(results, "A1.json", {"mock": False})
    _write(results, "A2.json", {"mock": False})
    _write(results, "A8.json", {"mock": False, "status": "blocked"})
    monkeypatch.setattr(build_readme, "RESULTS", results)

    body = build_readme.arms_block()

    assert "A8 | NOT YET RUN" in body
    assert "0.9" not in body
    assert "0.25" in body
