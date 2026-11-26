"""One cached LLM client for generation, judging and routing.

Cache key: provider + model + temperature + seed + max_tokens + messages hash.
Real calls record latency, token counts (when returned) and an estimated cost
from configs/prices.yaml. ``mock=True`` is deterministic and never touches the
network.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "litho_eval"
DEFAULT_PRICES = REPO_ROOT / "configs" / "prices.yaml"

Message = dict[str, str]


class MissingPriceError(RuntimeError):
    """Raised when configs/prices.yaml has no usable price for a model."""


def model_family(model: str) -> str:
    name = model.lower()
    for family in ("gpt-oss", "deepseek", "llama", "qwen", "mistral", "claude", "gemini", "phi"):
        if family in name:
            return family
    return name.split("/")[0]


def assert_different_family(judge_model: str, generator_model: str) -> None:
    if model_family(judge_model) == model_family(generator_model):
        raise ValueError(
            f"judge model {judge_model!r} is in the same family as the generator "
            f"under test ({generator_model!r})"
        )


def load_prices(path: str | Path = DEFAULT_PRICES) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get("models", {}) or {}


def _price_entry(prices: dict, model: str) -> dict:
    entry = prices.get(model)
    if entry is None:
        raise MissingPriceError(
            f"no price entry for {model!r} in configs/prices.yaml; add input_per_million "
            "and output_per_million before a real run"
        )
    for key in ("input_per_million", "output_per_million"):
        if entry.get(key) is None:
            raise MissingPriceError(
                f"{model!r} is missing {key} in configs/prices.yaml; fill it in"
            )
    return entry


def estimate_cost(
    prices: dict,
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> float:
    entry = _price_entry(prices, model)
    prompt = prompt_tokens or 0
    completion = completion_tokens or 0
    return (
        prompt / 1_000_000 * float(entry["input_per_million"])
        + completion / 1_000_000 * float(entry["output_per_million"])
    )


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    latency_s: float
    cost_usd: float
    cached: bool = False


def _normalise(messages: str | Sequence[Message]) -> list[Message]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return [dict(message) for message in messages]


def _messages_hash(messages: list[Message]) -> str:
    canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class LLMClient:
    def __init__(
        self,
        model: str,
        prices_path: str | Path = DEFAULT_PRICES,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 512,
        mock: bool = False,
        completer: Optional[Callable[[list[Message]], str]] = None,
    ) -> None:
        self.model = model
        self.provider = model.split("/")[0] if "/" in model else "litellm"
        self.prices_path = Path(prices_path)
        self.cache_dir = Path(cache_dir)
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.mock = mock
        self.completer = completer
        self._prices: dict | None = None

    def _cache_key(self, messages: list[Message], max_tokens: int) -> str:
        payload = json.dumps(
            {
                "provider": self.provider,
                "model": self.model,
                "temperature": self.temperature,
                "seed": self.seed,
                "max_tokens": max_tokens,
                "messages_sha256": _messages_hash(messages),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def complete(
        self,
        messages: str | Sequence[Message],
        max_tokens: int | None = None,
    ) -> LLMResponse:
        normalised = _normalise(messages)
        limit = max_tokens or self.max_tokens
        cache_file = self.cache_dir / f"{self._cache_key(normalised, limit)}.json"

        started = time.perf_counter()
        if cache_file.exists():
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            return LLMResponse(
                text=payload["text"],
                model=self.model,
                prompt_tokens=payload.get("prompt_tokens"),
                completion_tokens=payload.get("completion_tokens"),
                latency_s=time.perf_counter() - started,
                cost_usd=float(payload.get("cost_usd", 0.0)),
                cached=True,
            )

        if not self.mock:
            self._prices = self._prices or load_prices(self.prices_path)
            _price_entry(self._prices, self.model)

        text, prompt_tokens, completion_tokens = self._call(normalised, limit)
        latency = time.perf_counter() - started

        if self.mock:
            cost = 0.0
        else:
            self._prices = self._prices or load_prices(self.prices_path)
            cost = estimate_cost(self._prices, self.model, prompt_tokens, completion_tokens)

        response = LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=latency,
            cost_usd=cost,
            cached=False,
        )
        self._write_cache(cache_file, response)
        return response

    def _write_cache(self, cache_file: Path, response: LLMResponse) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "key": asdict(response),
            "text": response.text,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "cost_usd": response.cost_usd,
        }
        temp = cache_file.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temp, cache_file)

    def _call(self, messages: list[Message], max_tokens: int) -> tuple[str, int | None, int | None]:
        if self.completer is not None:
            text = self.completer(messages)
            return text, _estimate_tokens(messages), _count_tokens(text)

        if self.mock:
            text = _mock_response(messages)
            return text, _estimate_tokens(messages), _count_tokens(text)

        import litellm

        attempts = 6
        for attempt in range(attempts):
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    seed=self.seed,
                    max_tokens=max_tokens,
                )
                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
                completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
                return response.choices[0].message.content, prompt_tokens, completion_tokens
            except (litellm.RateLimitError, litellm.Timeout, litellm.APIConnectionError) as exc:
                if attempt == attempts - 1:
                    raise
                time.sleep(self._retry_delay(exc, attempt))

        raise RuntimeError("unreachable")

    @staticmethod
    def _retry_delay(exc: Exception, attempt: int) -> float:
        match = re.search(r"try again in ([0-9.]+)s", str(exc))
        suggested = float(match.group(1)) + 1.5 if match else 0.0
        return min(max(suggested, 5.0 * (2 ** attempt)), 65.0)


def _estimate_tokens(messages: list[Message]) -> int:
    text = " ".join(message.get("content", "") for message in messages)
    return len(text.split())


def _count_tokens(text: str) -> int:
    return len(text.split())


def _mock_response(messages: list[Message]) -> str:
    prompt = " ".join(message.get("content", "") for message in messages).lower()

    if "atomic factual claims" in prompt or '"claims"' in prompt:
        return json.dumps({"claims": ["Mock claim one.", "Mock claim two."]})
    if "supported" in prompt:
        return json.dumps({"supported": True, "reason": "mock"})
    if '"correct"' in prompt:
        return json.dumps({"correct": True, "reason": "mock"})

    digest = _messages_hash(messages)[:8]
    return f"MOCK::{digest}"
