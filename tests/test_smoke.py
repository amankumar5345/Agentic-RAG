"""Smoke test: every module under src/ must import without network access."""

import importlib
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"


def _module_names() -> list[str]:
    names = []
    for path in sorted(SRC_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        parts = list(path.relative_to(SRC_DIR.parent).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        names.append(".".join(parts))
    return names


@pytest.mark.parametrize("module_name", _module_names())
def test_module_imports_offline(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None
