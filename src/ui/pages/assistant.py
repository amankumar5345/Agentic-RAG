"""Assistant page: chat, trace panel, citations and geometry."""

from __future__ import annotations

import streamlit as st

from src.core.trace import arm_trace, load_traces, trace_for_question, trace_result
from src.ui.components.chat import render_chat
from src.ui.components.mock import MOCK_BANNER
from src.ui.data import NOT_RUN
from src.ui.state import setup_page


def render() -> None:
    setup_page("Assistant")
    st.title("Assistant")

    mode = st.session_state.get("mode", "demo")

    if mode == "demo":
        traces = load_traces()
        if not traces:
            st.warning(f"{NOT_RUN}: no recorded traces in results/traces/.")
            return
        if any(trace.get("mock") for trace in traces):
            st.warning(f"**{MOCK_BANNER}.** Demo replays recorded traces only.")
        questions = []
        for trace in traces:
            question = trace.get("question")
            if question and question not in questions:
                questions.append(question)
        if not questions:
            st.warning(f"{NOT_RUN}: recorded traces have no question field.")
            return
        st.caption(
            "Demo mode replays committed traces. Missing fields are shown as "
            "'not recorded'; nothing is synthesized."
        )
        selected = st.selectbox("Recorded question", questions)
        if st.button("Use this recorded question"):
            st.session_state["pending_question"] = selected

        def trace_fn(question: str) -> dict:
            return trace_result(trace_for_question(question, traces))

    else:
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
