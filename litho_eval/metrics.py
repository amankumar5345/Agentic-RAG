"""Retrieval, answer, faithfulness, abstention and cost metrics."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Sequence

from litho_eval.client import LLMClient, LLMResponse, assert_different_family
from litho_eval.stats import percentile

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JUDGE_PROMPT = REPO_ROOT / "configs" / "judge_prompt.md"
MATCH_THRESHOLD = 0.8

CLAIMS_PROMPT = Template(
    """Split the answer below into atomic factual claims.
Rules:
- Each claim must be a single, checkable statement.
- Do not add claims that are not in the answer.
- If the answer has no factual claims, return an empty list.
Return only JSON: {"claims": ["...", "..."]}

Answer:
$answer
"""
)

SUPPORT_PROMPT = Template(
    """You are checking one claim against retrieved context.
Return only JSON: {"supported": true, "reason": "one short sentence"}
If the context does not contain the claim, set supported to false.

Context:
$context

Claim:
$claim
"""
)


def tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(candidate: str, reference: str) -> float:
    """Token-level F1 between two spans."""
    candidate_tokens = Counter(tokenise(candidate))
    reference_tokens = Counter(tokenise(reference))
    if not candidate_tokens or not reference_tokens:
        return 0.0
    overlap = sum((candidate_tokens & reference_tokens).values())
    precision = overlap / sum(candidate_tokens.values())
    recall = overlap / sum(reference_tokens.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def retrieval_metrics(
    retrieved_texts: Sequence[str],
    gold_texts: Sequence[str],
    k: int = 5,
) -> dict:
    """Recall@k, hit@k and MRR using token-F1 >= 0.8 matching."""
    top_k = list(retrieved_texts[:k])
    if not gold_texts:
        return {"recall_at_k": None, "hit_at_k": None, "mrr": None, "k": k, "n_gold": 0}

    matched = 0
    first_rank = None
    for gold in gold_texts:
        best = max((token_f1(text, gold) for text in top_k), default=0.0)
        if best >= MATCH_THRESHOLD:
            matched += 1
    for rank, text in enumerate(top_k, start=1):
        if any(token_f1(text, gold) >= MATCH_THRESHOLD for gold in gold_texts):
            first_rank = rank
            break

    return {
        "recall_at_k": matched / len(gold_texts),
        "hit_at_k": 1.0 if matched else 0.0,
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "k": k,
        "n_gold": len(gold_texts),
    }


def numeric_match(answer: str, gold: str, tolerance: float | None) -> bool:
    numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", answer)
    gold_numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", gold)
    if not numbers or not gold_numbers:
        return False
    return abs(float(numbers[0]) - float(gold_numbers[0])) <= (tolerance or 0.0)


def parse_json(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in judge response: {text[:200]!r}")
    return json.loads(raw[start:end + 1])


@dataclass
class CorrectnessResult:
    correct: bool
    reason: str
    response: LLMResponse | None


@dataclass
class FaithfulnessResult:
    score: float
    claims: list[str]
    supported: list[bool]
    responses: list[LLMResponse]


class Judge:
    def __init__(
        self,
        client: LLMClient,
        judge_model: str,
        generator_model: str,
        prompt_path: str | Path = DEFAULT_JUDGE_PROMPT,
    ) -> None:
        assert_different_family(judge_model, generator_model)
        self.client = client
        self.model = judge_model
        self.rubric = Template(Path(prompt_path).read_text(encoding="utf-8"))

    def correctness(self, question: str, gold_answer: str, answer: str) -> CorrectnessResult:
        prompt = self.rubric.substitute(
            question=question,
            gold_answer=gold_answer,
            answer=answer,
        )
        response = self.client.complete(prompt)
        try:
            data = parse_json(response.text)
        except (ValueError, json.JSONDecodeError) as exc:
            return CorrectnessResult(False, f"judge parse error: {exc}", response)
        return CorrectnessResult(bool(data.get("correct")), str(data.get("reason", "")), response)

    def claims(self, answer: str) -> tuple[list[str], LLMResponse]:
        response = self.client.complete(CLAIMS_PROMPT.substitute(answer=answer))
        try:
            data = parse_json(response.text)
        except (ValueError, json.JSONDecodeError):
            return [], response
        claims = [str(claim) for claim in data.get("claims", []) if str(claim).strip()]
        return claims, response

    def supported(self, claim: str, context: str) -> tuple[bool, LLMResponse]:
        response = self.client.complete(
            SUPPORT_PROMPT.substitute(context=context, claim=claim)
        )
        try:
            data = parse_json(response.text)
        except (ValueError, json.JSONDecodeError):
            return False, response
        return bool(data.get("supported")), response


def faithfulness(judge: Judge, answer: str, contexts: Sequence[str]) -> FaithfulnessResult:
    claims, claim_response = judge.claims(answer)
    if not claims:
        return FaithfulnessResult(0.0, [], [], [claim_response])

    context = "\n\n".join(contexts)
    supported: list[bool] = []
    responses = [claim_response]
    for claim in claims:
        ok, response = judge.supported(claim, context)
        supported.append(ok)
        responses.append(response)

    return FaithfulnessResult(
        score=sum(supported) / len(supported),
        claims=claims,
        supported=supported,
        responses=responses,
    )


ABSTENTION_PHRASES = (
    "not answerable",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "insufficient information",
    "not enough information",
    "no information",
)


def is_abstention(text: str) -> bool:
    lowered = " ".join(text.lower().split())
    return not lowered or any(phrase in lowered for phrase in ABSTENTION_PHRASES)


def abstention_metrics(pairs: Sequence[tuple[bool, str]]) -> dict:
    """pairs: (is_unanswerable, output). Precision/recall over cases."""
    unanswerable = [output for flag, output in pairs if flag]
    predicted = [output for _, output in pairs if is_abstention(output)]
    correct = [output for flag, output in pairs if flag and is_abstention(output)]

    precision = len(correct) / len(predicted) if predicted else 0.0
    recall = len(correct) / len(unanswerable) if unanswerable else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "n_unanswerable": len(unanswerable),
        "n_predicted_abstention": len(predicted),
    }


def latency_summary(values: Sequence[float]) -> dict:
    if not values:
        return {"p50": None, "p95": None, "mean": None, "n": 0}
    return {
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "mean": sum(values) / len(values),
        "n": len(values),
    }


def cost_per_correct(total_cost_usd: float, correct_count: int) -> float | None:
    if correct_count <= 0:
        return None
    return total_cost_usd / correct_count
