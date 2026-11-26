"""Tests for scripts/claim_check.py."""

from pathlib import Path

from scripts.claim_check import check, check_text, result_numbers


def test_invented_decimal_is_flagged(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "guardrails.json").write_text('{"asr": 0.25}', encoding="utf-8")

    errors = check_text(Path("README.md"), "Accuracy: 0.93", {0.25})

    assert len(errors) == 1
    assert "0.93" in errors[0]


def test_matching_decimal_and_percent_pass(tmp_path):
    numbers = {0.25}

    assert check_text(Path("README.md"), "Accuracy: 0.25", numbers) == []
    assert check_text(Path("README.md"), "ASR: 25%", numbers) == []


def test_claims_file_is_exempt(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    claims = docs / "CLAIMS_TO_VERIFY.md"
    claims.write_text("Faithfulness >= 0.50", encoding="utf-8")

    assert check([docs], tmp_path / "results") == []


def test_code_fences_and_versions_are_ignored():
    numbers = set()

    assert check_text(Path("README.md"), "```\naccuracy 0.93\n```", numbers) == []
    assert check_text(Path("README.md"), "Python 3.12", numbers) == []


def test_mock_results_do_not_back_claims(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "guardrails.json").write_text(
        '{"mock": true, "asr": 0.93}', encoding="utf-8"
    )

    assert result_numbers(results) == set()

