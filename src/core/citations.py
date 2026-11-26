"""Citation formatting, validation and abstention."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

CITATION_RE = re.compile(r"\[C(\d+)\]")
PROCEDURAL_VERBS = (
    "use",
    "wear",
    "turn",
    "set",
    "calibrate",
    "clean",
    "check",
    "ensure",
    "avoid",
    "stop",
    "evacuate",
    "call",
    "open",
    "close",
    "adjust",
    "measure",
    "record",
    "verify",
    "replace",
    "connect",
    "disconnect",
    "vent",
    "purge",
    "do not",
    "never",
)
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
CITATION_RULE = (
    "Every factual or procedural claim must carry at least one [Cn] citation. "
    "Hazard-critical answers (chemicals, interlocks, emergency procedures) must "
    "also name the document and section."
)

ABSTENTION_TEXT = "I can't support this from the documents."


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    doc: str = ""
    section: str = ""

    def citation(self, snippet_chars: int = 240) -> dict:
        snippet = " ".join(self.text.split())
        return {
            "id": self.id,
            "doc": self.doc,
            "section": self.section,
            "snippet": snippet[:snippet_chars],
        }


def format_chunks(chunks: Sequence[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{chunk.id}] {chunk.doc} :: {chunk.section}\n{chunk.text}" for chunk in chunks
    )


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def is_claim_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(verb in lowered for verb in PROCEDURAL_VERBS) or bool(NUMBER_RE.search(sentence))


def citation_ids(text: str) -> list[str]:
    return [f"C{number}" for number in CITATION_RE.findall(text)]


@dataclass
class CitationValidation:
    ok: bool
    issues: list[str] = field(default_factory=list)
    citation_precision: float = 1.0
    coverage: float = 1.0
    supported: list[bool] = field(default_factory=list)
    cited_ids: list[str] = field(default_factory=list)


@dataclass
class CitationAnswer:
    text: str
    citations: list[dict]
    abstained: bool = False
    validation: CitationValidation | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "citations": self.citations,
            "abstained": self.abstained,
        }


def validate_citations(
    answer: str,
    chunks: Sequence[RetrievedChunk],
    judge_fn: Callable[[str, str], bool],
) -> CitationValidation:
    by_id = {chunk.id: chunk for chunk in chunks}
    sentences = split_sentences(answer)
    claim_sentences = [sentence for sentence in sentences if is_claim_sentence(sentence)]

    issues: list[str] = []
    supported: list[bool] = []
    cited_ids: list[str] = []
    covered_claims = 0

    for sentence in sentences:
        ids = citation_ids(sentence)
        if not ids:
            if sentence in claim_sentences:
                issues.append(f"missing citation: {sentence[:120]}")
            continue

        valid_ids = [citation_id for citation_id in ids if citation_id in by_id]
        for citation_id in ids:
            if citation_id not in by_id:
                issues.append(f"unknown citation {citation_id}")
            else:
                supported.append(bool(judge_fn(sentence, by_id[citation_id].text)))
                cited_ids.append(citation_id)

        if sentence in claim_sentences and valid_ids:
            covered_claims += 1
        elif sentence in claim_sentences:
            issues.append(f"citation does not exist for claim: {sentence[:120]}")

    for sentence in claim_sentences:
        if not citation_ids(sentence):
            issues.append(f"claim has no citation: {sentence[:120]}")

    if cited_ids:
        precision = sum(1 for value in supported if value) / len(supported)
    else:
        precision = 0.0 if claim_sentences else 1.0

    coverage = (covered_claims / len(claim_sentences)) if claim_sentences else 1.0
    if any(not value for value in supported):
        issues.append("one or more citations do not support their sentence")

    ok = not issues
    return CitationValidation(
        ok=ok,
        issues=list(dict.fromkeys(issues)),
        citation_precision=precision,
        coverage=coverage,
        supported=supported,
        cited_ids=cited_ids,
    )


def citations_used(
    answer: str,
    chunks: Sequence[RetrievedChunk],
) -> list[dict]:
    by_id = {chunk.id: chunk for chunk in chunks}
    usage = []
    seen = set()
    for citation_id in citation_ids(answer):
        if citation_id in seen or citation_id not in by_id:
            continue
        seen.add(citation_id)
        usage.append(by_id[citation_id].citation())
    return usage


def closest_sources(chunks: Sequence[RetrievedChunk], limit: int = 3) -> list[dict]:
    return [chunk.citation() for chunk in chunks[:limit]]


def enforce_citations(
    question: str,
    chunks: Sequence[RetrievedChunk],
    answer_fn: Callable[[str, str, str | None], str],
    judge_fn: Callable[[str, str], bool],
    max_repairs: int = 1,
) -> CitationAnswer:
    context = format_chunks(chunks)
    answer = answer_fn(question, context, None)
    validation = validate_citations(answer, chunks, judge_fn)

    for _ in range(max_repairs):
        if validation.ok:
            break
        feedback = "; ".join(validation.issues)
        answer = answer_fn(question, context, feedback)
        validation = validate_citations(answer, chunks, judge_fn)

    if validation.ok:
        return CitationAnswer(
            text=answer,
            citations=citations_used(answer, chunks),
            abstained=False,
            validation=validation,
        )

    sources = closest_sources(chunks)
    text = ABSTENTION_TEXT
    if sources:
        text += " Closest sources: " + "; ".join(
            f"[{source['id']}] {source['doc']} / {source['section']}"
            for source in sources
        )
    return CitationAnswer(
        text=text,
        citations=sources,
        abstained=True,
        validation=validation,
    )
