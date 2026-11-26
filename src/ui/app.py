"""Navigation only. Every page owns its content and setup."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

pages = [
    st.Page("pages/methodology.py", title="Methodology", icon=":material/menu_book:"),
    st.Page("pages/benchmark.py", title="Benchmark", icon=":material/bar_chart:"),
    st.Page("pages/dataset_explorer.py", title="Dataset Explorer", icon=":material/table:"),
    st.Page("pages/geometry_lab.py", title="Geometry Lab", icon=":material/polyline:"),
    st.Page("pages/safety_lab.py", title="Safety Lab", icon=":material/shield:"),
    st.Page("pages/playground.py", title="Arm Playground", icon=":material/science:"),
    st.Page("pages/assistant.py", title="Assistant", icon=":material/chat:"),
]

st.navigation(pages).run()
