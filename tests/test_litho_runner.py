"""Mock-mode tests for the runner and comparison."""

import hashlib
import json
from datetime import datetime, timezone

import pytest

import litho_eval.runner as runner
from bench.schema import GoldSource, QACase, write_cases
from litho_eval import compare as compare_mod

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
CORPUS = "Laser safety requires goggles.\n\nThe dose is 42 mJ."


def source_for(span: str) -> GoldSource:
    start = CORPUS.index(span)
    return GoldSource(
        doc="doc.pdf",
        section="s",
        char_start=start,
        char_end=start + len(span),
        text_hash=hashlib.sha256(span.encode()).hexdigest(),
    )


def make_case(case_id, question, answer, subtype="single_hop", sources=None, tolerance=None):
    return QACase(
        id=case_id,
        split="test",
        origin="human",
        status="verified",
        reviewer="tester",
        reviewed_at=NOW,
        question=question,
        gold_answer=answer,
        gold_sources=sources or [],
        subtype=subtype,
        numeric_tolerance=tolerance,
    )


def build_dataset(tmp_path):
    data_dir = tmp_path / "data"
    cases = [
        make_case(
            "qa-1",
            "What does laser safety require?",
            "Laser safety requires goggles.",
            sources=[source_for("Laser safety requires goggles.")],
        ),
        make_case(
            "qa-2",
            "What dose is used?",
            "42 mJ",
            subtype="numeric",
            sources=[source_for("42 mJ")],
            tolerance=0.1,
        ),
        make_case(
            "qa-3",
            "What is the plutonium wafer diameter?",
            "Not answerable from the provided corpus.",
            subtype="unanswerable",
            sources=[],
        ),
    ]
    write_cases(cases, data_dir)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(CORPUS, encoding="utf-8")
    return data_dir, corpus


def run_args(tmp_path, data_dir, corpus, out, *extra):
    return [
        "--arm",
        "A4",
        "--mock",
        "--split",
        "test",
        "--data-dir",
        str(data_dir),
        "--corpus",
        str(corpus),
        "--out",
        str(out),
        "--cache-dir",
        str(tmp_path / "cache"),
        *extra,
    ]


def test_runner_mock_mode(tmp_path):
    data_dir, corpus = build_dataset(tmp_path)
    out = tmp_path / "A4.json"

    assert runner.main(run_args(tmp_path, data_dir, corpus, out)) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "1.0"
    assert payload["arm"] == "A4"
    assert payload["split"] == "test"
    assert payload["seed"] == 42
    assert payload["models"] == {
        "generator": "openai/gpt-oss-120b",
        "judge": "groq/qwen/qwen3.8-27b",
    }
    assert len(payload["git_commit"]) >= 7
    assert len(payload["config_hash"]) == 64
    assert len(payload["dataset_hash"]) == 64
    assert payload["timestamp"]

    aggregates = payload["aggregates"]
    assert aggregates["n_cases"] == 3
    assert aggregates["n_answerable"] == 2
    assert aggregates["n_unanswerable"] == 1
    assert aggregates["answer_correctness"] == 1.0
    assert aggregates["faithfulness"] == 1.0
    assert aggregates["retrieval"]["recall_at_k"] == 1.0
    assert aggregates["abstention"]["precision"] == 1.0
    assert aggregates["abstention"]["recall"] == 1.0
    assert aggregates["cost_usd"]["total"] == 0.0

    record = payload["cases"][0]
    assert record["input"]
    assert record["output"]
    assert record["retrieved"]
    assert set(record["latencies"]) == {
        "retrieval_s",
        "generation_s",
        "judge_s",
        "total_s",
    }


def test_resume_skips_cached_cases(tmp_path, monkeypatch):
    data_dir, corpus = build_dataset(tmp_path)
    out = tmp_path / "A4.json"
    calls = []
    original = runner.evaluate_case

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "evaluate_case", counting)

    assert runner.main(run_args(tmp_path, data_dir, corpus, out)) == 0
    assert len(calls) == 3

    assert runner.main(run_args(tmp_path, data_dir, corpus, out, "--resume")) == 0
    assert len(calls) == 3


def test_resume_rejects_changed_config(tmp_path):
    data_dir, corpus = build_dataset(tmp_path)
    out = tmp_path / "A4.json"

    runner.main(run_args(tmp_path, data_dir, corpus, out))

    with pytest.raises(SystemExit):
        runner.main(run_args(tmp_path, data_dir, corpus, out, "--resume", "--top-k", "3"))


def _record(case_id, correct, score=1.0):
    return {
        "id": case_id,
        "correctness": {"correct": correct},
        "faithfulness": {"score": score},
        "retrieval": {"recall_at_k": score, "mrr": score},
        "latencies": {"total_s": 0.1},
        "cost_usd": 0.01,
    }


def test_load_results_skips_non_arm_files(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "A0.json").write_text(json.dumps({"arm": "A0"}), encoding="utf-8")
    (results / "injection.json").write_text(json.dumps({"mock": True}), encoding="utf-8")

    assert set(compare_mod.load_results(results)) == {"A0"}


def test_compare_paired_delta(tmp_path):
    results = tmp_path / "results"
    results.mkdir()

    baseline = {
        "arm": "A0",
        "dataset_hash": "dataset",
        "split": "test",
        "mock": True,
        "cases": [_record("1", False), _record("2", False)],
    }
    arm = {
        "arm": "A4",
        "dataset_hash": "dataset",
        "split": "test",
        "mock": True,
        "cases": [_record("1", True), _record("2", True)],
    }
    (results / "A0.json").write_text(json.dumps(baseline), encoding="utf-8")
    (results / "A4.json").write_text(json.dumps(arm), encoding="utf-8")

    payload = compare_mod.compare(compare_mod.load_results(results), "A0")
    metric = payload["arms"]["A4"]["metrics"]["answer_correctness"]

    assert metric["delta"] == 1.0
    assert metric["ci95"] == [1.0, 1.0]
    assert "A4" in compare_mod.markdown(payload)
