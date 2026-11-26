"""Assistant page: chat, trace panel, citations and geometry."""

from __future__ import annotations

import streamlit as st

from src.core.trace import demo_trace
from src.ui.components.chat import render_chat
from src.ui.data import NOT_RUN, load_arm
from src.ui.state import setup_page


def render() -> None:
    setup_page("Assistant")
    st.title("Assistant")

    mode = st.session_state.get("mode", "demo")
    payload = load_arm("A5")
    cases = [case for case in (payload or {}).get("cases", []) if case.get("type") == "qa"]
    by_question = {case.get("input"): case for case in cases}

    if mode == "demo":
        if not cases:
            st.warning(f"{NOT_RUN}: no recorded A5 answers.")
            return
        st.caption("Demo mode replays recorded A5 answers. Pick one, then ask it in the chat box.")
        options = [case.get("input") for case in cases]
        selected = st.selectbox("Recorded question", options)
        if st.button("Use this recorded question"):
            st.session_state["pending_question"] = selected

        def trace_fn(question: str) -> dict:
            case = by_question.get(question)
            if case is None:
                return {
                    "answer": "No recorded trace for this question in demo mode.",
                    "trace": None,
                    "geometry": None,
                }
            return demo_trace(case)

    else:
        from src.core.trace import arm_trace
        from src.ui.data import get_live_arm

        def trace_fn(question: str) -> dict:
            try:
                return arm_trace(question, get_live_arm("A8"))
            except Exception as exc:
                return {
                    "answer": f"Live A8 run failed: {exc}",
                    "trace": None,
                    "geometry": None,
                }

    render_chat(trace_fn)


if __name__ == "__main__":
    render()
