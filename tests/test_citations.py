"""Tests for citation validation, repair and abstention."""

from src.core.citations import (
    ABSTENTION_TEXT,
    RetrievedChunk,
    citation_ids,
    citations_used,
    enforce_citations,
    format_chunks,
    validate_citations,
)
from src.core.retriever import retrieve_chunks

CHUNK = RetrievedChunk(
    id="C1",
    text="Wear safety goggles when using the laser.",
    doc="laser_safety.pdf",
    section="PPE",
)


def always_supported(_sentence, _chunk):
    return True


def never_supported(_sentence, _chunk):
    return False


def test_correct_answer_passes():
    validation = validate_citations(
        "Wear safety goggles when using the laser [C1].",
        [CHUNK],
        always_supported,
    )

    assert validation.ok
    assert validation.coverage == 1.0
    assert validation.citation_precision == 1.0


def test_missing_citation_is_flagged():
    validation = validate_citations(
        "Wear safety goggles when using the laser.",
        [CHUNK],
        always_supported,
    )

    assert not validation.ok
    assert any("missing citation" in issue or "no citation" in issue for issue in validation.issues)


def test_bogus_citation_is_flagged():
    validation = validate_citations(
        "Wear safety goggles when using the laser [C9].",
        [CHUNK],
        always_supported,
    )

    assert not validation.ok
    assert any("unknown citation C9" in issue for issue in validation.issues)


def test_unsupported_claim_is_flagged():
    validation = validate_citations(
        "Wear safety goggles when using the laser [C1].",
        [CHUNK],
        never_supported,
    )

    assert not validation.ok
    assert validation.citation_precision == 0.0
    assert any("do not support" in issue for issue in validation.issues)


def test_repair_retry_returns_supported_answer():
    answers = iter(
        [
            "Wear safety goggles when using the laser.",
            "Wear safety goggles when using the laser [C1].",
        ]
    )

    result = enforce_citations(
        "What PPE is needed?",
        [CHUNK],
        lambda question, context, feedback: next(answers),
        always_supported,
    )

    assert result.abstained is False
    assert "[C1]" in result.text
    assert result.to_dict()["citations"][0]["id"] == "C1"


def test_failed_repair_abstains_with_sources():
    result = enforce_citations(
        "What PPE is needed?",
        [CHUNK],
        lambda question, context, feedback: "Wear safety goggles when using the laser.",
        always_supported,
    )

    assert result.abstained is True
    assert result.text.startswith(ABSTENTION_TEXT)
    assert result.to_dict()["citations"][0]["doc"] == "laser_safety.pdf"


def test_format_and_extraction():
    assert "[C1] laser_safety.pdf :: PPE" in format_chunks([CHUNK])
    assert citation_ids("A [C1] and B [C2] and [C1].") == ["C1", "C2", "C1"]
    assert [citation["id"] for citation in citations_used("[C1] and [C1]", [CHUNK])] == ["C1"]


class FakeDocument:
    def __init__(self, text, source, section):
        self.page_content = text
        self.metadata = {"source": source, "section": section}


class FakeRetriever:
    def invoke(self, question):
        return [
            FakeDocument("alpha", r"docs\a.pdf", "Intro"),
            FakeDocument("beta", "docs/b.pdf", "Process"),
        ]


def test_retrieve_chunks_assigns_stable_ids_and_metadata():
    chunks = retrieve_chunks("question", retriever=FakeRetriever())

    assert [chunk.id for chunk in chunks] == ["C1", "C2"]
    assert chunks[0].doc == "a.pdf"
    assert chunks[1].section == "Process"
