"""Arm Playground: compare recorded arm outputs on the same question."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from litho_eval.metrics import token_f1
from src.ui.components.citations import render_citations
from src.ui.data import NOT_RUN, load_arm, load_cases
from src.ui.state import setup_page


def _arm_cases() -> dict[str, dict]:
    cases = {}
    for index in range(9):
        arm = f"A{index}"
        payload = load_arm(arm)
        if payload:
            cases[arm] = {case["id"]: case for case in payload.get("cases", [])}
    return cases


def render() -> None:
    setup_page("Arm Playground")
    st.title("Arm Playground")

    mode = st.session_state.get("mode", "demo")
    if mode == "live":
        from src.ui.data import get_live_arm

        st.caption("Live mode: each selected arm runs with real clients.")
        question = st.text_input("Question")
        available = [f"A{index}" for index in range(9)]
        selected_arms = st.multiselect("Arms (up to three)", available, default=available[:3])
        selected_arms = selected_arms[:3]
        if st.button("Run live arms") and question.strip() and selected_arms:
            columns = st.columns(len(selected_arms))
            for column, arm in zip(columns, selected_arms):
                with column:
                    st.subheader(arm)
                    try:
                        output = get_live_arm(arm).answer(question)
                    except Exception as exc:
                        st.error(f"{arm} failed: {exc}")
                        continue
                    st.markdown(output.answer)
                    if output.retrieved:
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {
                                        "id": chunk.id,
                                        "doc": chunk.doc,
                                        "section": chunk.section,
                                    }
                                    for chunk in output.retrieved
                                ]
                            ),
                            width="stretch",
                        )
                    st.write(
                        {
                            "retrieval_calls": output.retrieval_calls,
                            "latency_s": output.stage_latencies.get("total_s"),
                            "cost_usd": output.cost_usd,
                        }
                    )
                    render_citations(output.citations, f"live-{arm}")
        return

    arms = _arm_cases()
    if not arms:
        st.warning(f"{NOT_RUN}: no arm results.")
        return

    available = sorted(arms)
    selected_arms = st.multiselect("Arms (up to three)", available, default=available[:3])
    selected_arms = selected_arms[:3]

    common = set.intersection(*(set(arms[arm]) for arm in selected_arms)) if selected_arms else set()
    if not common:
        st.info("No common cases across the selected arms.")
        return

    cases = {case["id"]: case for case in load_cases() if case.get("type") == "qa"}
    options = sorted(common, key=lambda case_id: cases.get(case_id, {}).get("question", case_id))
    labels = {
        case_id: cases.get(case_id, {}).get("question", case_id) for case_id in options
    }
    selected_id = st.selectbox("Question", options, format_func=lambda key: labels[key])

    columns = st.columns(len(selected_arms))
    for column, arm in zip(columns, selected_arms):
        case = arms[arm].get(selected_id)
        with column:
            st.subheader(arm)
            if not case:
                st.info(NOT_RUN)
                continue
            if case.get("mock"):
                st.caption("MOCK recorded output")
            st.markdown(case.get("output", ""))
            retrieved = case.get("retrieved", [])
            if retrieved:
                st.dataframe(pd.DataFrame(retrieved), width="stretch")
                gold = cases.get(selected_id, {}).get("gold_answer", "")
                matches = [
                    token_f1(item.get("text", ""), gold) >= 0.8 for item in retrieved
                ]
                st.caption(
                    "Gold source matches: "
                    + ", ".join(
                        f"C{index}" for index, match in enumerate(matches, start=1) if match
                    )
                )
            latencies = case.get("latencies", {})
            st.write({"latency_s": latencies.get("total_s"), "cost_usd": case.get("cost_usd")})
            render_citations(case.get("citations"), f"playground-{arm}")


if __name__ == "__main__":
    render()
