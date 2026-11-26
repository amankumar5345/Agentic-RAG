"""Browse LithoBench and injection cases."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ui.data import (
    derived_difficulty,
    gold_text,
    load_cases,
    load_doc,
    load_injection_cases,
)
from src.ui.state import setup_page


def render() -> None:
    setup_page("Dataset Explorer")
    st.title("Dataset Explorer")

    cases = load_cases()
    if not cases:
        st.warning("NOT YET RUN: no LithoBench cases found.")
        return

    frame = pd.DataFrame(
        [
            {
                "id": case.get("id"),
                "type": case.get("type"),
                "subtype": case.get("subtype") or case.get("category") or "",
                "split": case.get("split"),
                "origin": case.get("origin"),
                "status": case.get("status"),
                "difficulty": derived_difficulty(case),
            }
            for case in cases
        ]
    )

    st.sidebar.subheader("Filters")
    types = st.sidebar.multiselect("Type", sorted(frame["type"].dropna().unique()))
    splits = st.sidebar.multiselect("Split", sorted(frame["split"].dropna().unique()))
    origins = st.sidebar.multiselect("Origin", sorted(frame["origin"].dropna().unique()))
    statuses = st.sidebar.multiselect("Status", sorted(frame["status"].dropna().unique()))
    difficulties = st.sidebar.multiselect(
        "Difficulty (derived)", sorted(frame["difficulty"].dropna().unique())
    )

    filtered = frame.copy()
    for column, values in (
        ("type", types),
        ("split", splits),
        ("origin", origins),
        ("status", statuses),
        ("difficulty", difficulties),
    ):
        if values:
            filtered = filtered[filtered[column].isin(values)]

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Counts")
        counts = (
            filtered.groupby(["type", "subtype"]).size().reset_index(name="count")
            if not filtered.empty
            else pd.DataFrame(columns=["type", "subtype", "count"])
        )
        st.dataframe(counts, width="stretch")
        st.caption(f"{len(filtered)} cases after filters")
    with right:
        st.subheader("Cases")
        if filtered.empty:
            st.info("No cases match the filters.")
        else:
            selected_id = st.selectbox("Case", filtered["id"].tolist())
            case = next(case for case in cases if case.get("id") == selected_id)
            st.json(case)
            if case.get("type") == "qa":
                st.markdown("**Gold answer**")
                st.write(case.get("gold_answer"))
                st.markdown("**Gold source**")
                st.text(gold_text(case)[:4000] or "No source text available.")

    st.subheader("Dataset card")
    st.markdown(load_doc("bench/DATASET_CARD.md") or "NOT YET RUN")

    st.divider()
    st.subheader("Injection cases")
    injection = load_injection_cases()
    if not injection:
        st.info("NOT YET RUN: no injection cases.")
        return
    injection_frame = pd.DataFrame(
        [
            {
                "id": case.get("id"),
                "attack_type": case.get("attack_type"),
                "split": case.get("split"),
                "status": case.get("status"),
                "question": case.get("question"),
            }
            for case in injection
        ]
    )
    st.dataframe(injection_frame, width="stretch")


if __name__ == "__main__":
    render()

