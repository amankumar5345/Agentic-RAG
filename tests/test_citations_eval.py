"""A8 evidence must be real; blocked A8 keeps citations blocked."""

from __future__ import annotations

import json

from litho_eval.citations_eval import main


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_blocked_a8_keeps_citations_blocked(tmp_path):
    a5 = tmp_path / "A5.json"
    a8 = tmp_path / "A8.json"
    out = tmp_path / "citations.json"
    _write(a5, {"schema_version": "1.0", "mock": False, "cases": []})
    _write(a8, {"schema_version": "1.0", "mock": False, "status": "blocked"})

    code = main(["--a5", str(a5), "--a8", str(a8), "--out", str(out)])

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert code == 2
    assert payload["status"] == "blocked"
    assert "NOT YET RUN" in payload["reason"]
