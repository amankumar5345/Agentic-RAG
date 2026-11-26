"""Hand-made metric tests."""

import pytest

from litho_eval.client import LLMClient
from litho_eval.metrics import (
    Judge,
    abstention_metrics,
    cost_per_correct,
    faithfulness,
    is_abstention,
    latency_summary,
    numeric_match,
    parse_json,
    retrieval_metrics,
    token_f1,
)


def test_token_f1():
    assert token_f1("Alpha  beta!", "alpha beta") == 1.0
    assert token_f1("alpha", "alpha beta") == pytest.approx(2 / 3)
    assert token_f1("", "alpha") == 0.0


def test_retrieval_metrics_exact_match():
    metrics = retrieval_metrics(["alpha beta gamma", "unrelated"], ["alpha beta gamma"], k=5)

    assert metrics["recall_at_k"] == 1.0
    assert metrics["hit_at_k"] == 1.0
    assert metrics["mrr"] == 1.0


def test_retrieval_metrics_mrr_second_position():
    metrics = retrieval_metrics(
        ["totally different words", "alpha beta gamma"],
        ["alpha beta gamma"],
        k=5,
    )

    assert metrics["mrr"] == 0.5


def test_retrieval_metrics_partial_recall_two_golds():
    metrics = retrieval_metrics(
        ["alpha beta gamma"],
        ["alpha beta gamma", "delta epsilon zeta"],
        k=5,
    )

    assert metrics["recall_at_k"] == 0.5
    assert metrics["hit_at_k"] == 1.0


def test_retrieval_metrics_without_gold():
    metrics = retrieval_metrics(["anything"], [], k=5)
    assert metrics["recall_at_k"] is None


def test_numeric_match_tolerance():
    assert numeric_match("The answer is 42.0 mJ", "42", 0.1)
    assert not numeric_match("43", "42", 0.5)
    assert not numeric_match("no number", "42", 0.5)


class FakeJudge:
    def claims(self, answer):
        return ["supported claim", "unsupported claim"], "claims-response"

    def supported(self, claim, context):
        return claim == "supported claim", "support-response"


def test_faithfulness_hand_score():
    result = faithfulness(FakeJudge(), "answer", ["context"])
    assert result.score == 0.5
    assert result.supported == [True, False]


def test_judge_in_mock_mode(tmp_path):
    judge = Judge(
        LLMClient("deepseek/deepseek-chat", cache_dir=tmp_path, mock=True),
        "deepseek/deepseek-chat",
        "openai/gpt-oss-120b",
    )

    correctness = judge.correctness("q", "g", "a")
    assert correctness.correct is True

    result = faithfulness(judge, "answer", ["context"])
    assert result.score == 1.0


def test_abstention_metrics_hand_values():
    result = abstention_metrics(
        [
            (True, "Not answerable from the provided corpus."),
            (True, "42"),
            (False, "42"),
        ]
    )

    assert result["precision"] == 1.0
    assert result["recall"] == 0.5


def test_is_abstention():
    assert is_abstention("")
    assert is_abstention("I cannot answer that.")
    assert not is_abstention("42")


def test_latency_summary():
    summary = latency_summary([0.0, 10.0])
    assert summary["p50"] == 5.0
    assert summary["p95"] == 9.5


def test_cost_per_correct():
    assert cost_per_correct(10.0, 4) == 2.5
    assert cost_per_correct(10.0, 0) is None


def test_parse_json_with_fence():
    assert parse_json('```json\n{"correct": true}\n```') == {"correct": True}
