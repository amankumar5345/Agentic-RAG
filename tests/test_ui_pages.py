"""AppTest smoke tests for every page in demo mode."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGES = {
    "methodology": "src/ui/pages/methodology.py",
    "benchmark": "src/ui/pages/benchmark.py",
    "dataset": "src/ui/pages/dataset_explorer.py",
    "geometry": "src/ui/pages/geometry_lab.py",
    "safety": "src/ui/pages/safety_lab.py",
    "playground": "src/ui/pages/playground.py",
    "assistant": "src/ui/pages/assistant.py",
}


def _run(name):
    at = AppTest.from_file(str(REPO_ROOT / PAGES[name]), default_timeout=30).run()
    assert not at.exception, [str(error.value) for error in at.exception]
    return at


def test_methodology_renders():
    at = _run("methodology")
    assert at.title[0].value == "Methodology"


def test_benchmark_renders():
    at = _run("benchmark")
    assert at.title[0].value == "Benchmark"
    assert len(at.selectbox) >= 1 or len(at.dataframe) >= 1


def test_dataset_explorer_renders():
    at = _run("dataset")
    assert at.title[0].value == "Dataset Explorer"
    assert len(at.selectbox) >= 1


def test_geometry_lab_renders():
    at = _run("geometry")
    assert at.title[0].value == "Geometry Lab"
    assert len(at.selectbox) >= 1
    assert len(at.button) >= 1


def test_safety_lab_renders():
    at = _run("safety")
    assert at.title[0].value == "Safety Lab"
    assert len(at.radio) >= 1


def test_playground_renders():
    at = _run("playground")
    assert at.title[0].value == "Arm Playground"
    assert len(at.multiselect) >= 1


def test_assistant_renders():
    at = _run("assistant")
    assert at.title[0].value == "Assistant"
    assert len(at.chat_input) >= 1
    assert len(at.selectbox) >= 1


def test_app_navigation_renders():
    at = AppTest.from_file(str(REPO_ROOT / "src/ui/app.py"), default_timeout=30).run()
    assert not at.exception, [str(error.value) for error in at.exception]
    assert at.title[0].value == "Methodology"


def _warnings(at) -> str:
    return " ".join(warning.value for warning in at.warning)


def test_mock_pages_show_the_page_level_banner():
    for page in ("benchmark", "safety", "geometry", "playground", "assistant"):
        at = _run(page)
        assert "Contains mock results, not measured" in _warnings(at), page
