"""Assemble the project handbook HTML and render it to PDF with headless Edge.

Usage:
    python docs/guide/build_pdf.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

GUIDE_DIR = Path(__file__).resolve().parent
REPO_ROOT = GUIDE_DIR.parents[1]
STYLE = GUIDE_DIR / "style.css"
PARTS = [GUIDE_DIR / f"part{i}.html" for i in range(1, 5)]
HTML_OUT = REPO_ROOT / "docs" / "Litho_Agentic_RAG_Project_Guide.html"
PDF_OUT = REPO_ROOT / "docs" / "Litho_Agentic_RAG_Project_Guide.pdf"

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]


def find_browser() -> str:
    for candidate in EDGE_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    for name in ("msedge", "chrome", "google-chrome", "chromium"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit("No Chromium-based browser found for PDF rendering.")


def build_html() -> None:
    css = STYLE.read_text(encoding="utf-8")
    body = "\n".join(part.read_text(encoding="utf-8") for part in PARTS)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Litho Agentic RAG - Project Handbook</title>
<meta name="author" content="Agentic-RAG project">
<meta name="description" content="Purpose, architecture, workflow, evaluation and interview preparation for the Litho Agentic RAG project.">
<style>
{css}
</style>
</head>
<body>
{body}
</body>
</html>
"""
    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"wrote {HTML_OUT} ({HTML_OUT.stat().st_size:,} bytes)")


def render_pdf(browser: str) -> None:
    if PDF_OUT.exists():
        PDF_OUT.unlink()
    with tempfile.TemporaryDirectory(prefix="litho_pdf_") as user_data:
        command = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=15000",
            f"--user-data-dir={user_data}",
            f"--print-to-pdf={PDF_OUT}",
            HTML_OUT.as_uri(),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if not PDF_OUT.exists():
        print(result.stdout)
        print(result.stderr)
        raise SystemExit("PDF was not produced.")


def verify_pdf() -> None:
    data = PDF_OUT.read_bytes()
    if not data.startswith(b"%PDF"):
        raise SystemExit("Output is not a PDF.")
    pages = data.count(b"/Type /Page") - data.count(b"/Type /Pages")
    print(f"wrote {PDF_OUT} ({len(data):,} bytes, about {pages} pages)")


def main() -> int:
    missing = [str(part) for part in PARTS if not part.exists()]
    if missing:
        raise SystemExit(f"missing parts: {missing}")
    build_html()
    render_pdf(find_browser())
    verify_pdf()
    return 0


if __name__ == "__main__":
    sys.exit(main())
