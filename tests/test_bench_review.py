"""Offline tests for the review decisions."""

import hashlib
from datetime import datetime, timezone

from bench.review import apply_decision, source_text
from bench.schema import GoldSource, QACase

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_case(text="0123456789", stored_hash=None):
    return QACase(
        id="qa-1",
        split="test",
        origin="llm_draft",
        status="draft",
        question="what?",
        gold_answer="answer",
        gold_sources=[
            GoldSource(
                doc="a.pdf",
                section="s",
                char_start=0,
                char_end=len(text),
                text_hash=stored_hash or hashlib.sha256(text.encode()).hexdigest(),
            )
        ],
        subtype="single_hop",
    )


def test_accept_verifies_and_keeps_llm_draft_origin():
    updated = apply_decision(make_case(), "accept", "alice", now=NOW)

    assert updated.status == "verified"
    assert updated.reviewer == "alice"
    assert updated.reviewed_at == NOW
    assert updated.origin == "llm_draft"


def test_edit_marks_llm_edited():
    case = make_case()
    edited = case.model_copy(update={"gold_answer": "corrected"})

    updated = apply_decision(case, "edit", "bob", now=NOW, edited=edited)

    assert updated.status == "verified"
    assert updated.origin == "llm_edited"
    assert updated.gold_answer == "corrected"


def test_reject_returns_none_and_skip_is_unchanged():
    case = make_case()

    assert apply_decision(case, "reject", "bob", now=NOW) is None
    assert apply_decision(case, "skip", "bob", now=NOW) is case


def test_source_text_reports_hash_mismatch():
    corpus = "0123456789"

    assert "[WARNING: text_hash mismatch]" not in source_text(make_case(), corpus)
    assert "[WARNING: text_hash mismatch]" in source_text(make_case(stored_hash="wrong"), corpus)
