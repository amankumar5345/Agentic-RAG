"""Tests for the shared cached LLM client."""

import pytest

from litho_eval.client import (
    LLMClient,
    MissingPriceError,
    assert_different_family,
    estimate_cost,
)


def test_second_call_is_a_cache_hit(tmp_path):
    calls = []

    def completer(messages):
        calls.append(messages)
        return "hello"

    client = LLMClient(
        "test/model",
        cache_dir=tmp_path,
        mock=True,
        completer=completer,
    )

    first = client.complete("same prompt")
    second = client.complete("same prompt")

    assert first.cached is False
    assert second.cached is True
    assert first.text == second.text == "hello"
    assert len(calls) == 1


def test_cache_key_depends_on_seed(tmp_path):
    calls = []

    def completer(messages):
        calls.append(messages)
        return "x"

    LLMClient("test/model", cache_dir=tmp_path, seed=1, mock=True, completer=completer).complete("p")
    LLMClient("test/model", cache_dir=tmp_path, seed=2, mock=True, completer=completer).complete("p")

    assert len(calls) == 2


def test_missing_price_fails_before_the_call(tmp_path):
    prices = tmp_path / "prices.yaml"
    prices.write_text("models: {}\n", encoding="utf-8")
    calls = []

    client = LLMClient(
        "openai/gpt-oss-120b",
        prices_path=prices,
        cache_dir=tmp_path / "cache",
        mock=False,
        completer=lambda messages: calls.append(messages) or "x",
    )

    with pytest.raises(MissingPriceError):
        client.complete("hi")
    assert calls == []


def test_estimate_cost():
    prices = {
        "model": {"input_per_million": 1.0, "output_per_million": 2.0},
    }
    assert estimate_cost(prices, "model", 1_000_000, 1_000_000) == 3.0
    assert estimate_cost(prices, "model", None, None) == 0.0


def test_judge_family_must_differ():
    with pytest.raises(ValueError):
        assert_different_family("openai/gpt-oss-20b", "openai/gpt-oss-120b")

    assert_different_family("groq/qwen/qwen3.8-27b", "openai/gpt-oss-120b")


def test_mock_mode_is_deterministic(tmp_path):
    first = LLMClient("test/model", cache_dir=tmp_path, mock=True).complete("p")
    second = LLMClient("test/model", cache_dir=tmp_path, mock=True).complete("p")

    assert first.text == second.text
    assert second.cached is True
    assert first.cost_usd == 0.0
