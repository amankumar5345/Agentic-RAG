"""Geometry Lab: commands, validation, preview, send and latency."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.core.geometry import replay
from src.ui.components.charts import latency_timeline
from src.ui.components.geometry_card import render_geometry_card
from src.ui.data import NOT_RUN, load_cases, load_result
from src.ui.state import setup_page


def render() -> None:
    setup_page("Geometry Lab")
    st.title("Geometry Lab")

    geometry_cases = [case for case in load_cases() if case.get("type") == "geometry"]
    if not geometry_cases:
        st.warning(f"{NOT_RUN}: no geometry cases.")
        return

    ids = [case["id"] for case in geometry_cases]
    selected_id = st.selectbox("LithoBench case", ids)
    case = next(case for case in geometry_cases if case["id"] == selected_id)

    result = load_result("geometry")
    result_case = None
    if result:
        result_case = next(
            (item for item in result.get("cases", []) if item.get("id") == selected_id),
            None,
        )

    default_commands = (
        result_case.get("prediction") if result_case and result_case.get("prediction") else case["gold_commands"]
    )
    raw = st.text_area("Commands JSON", json.dumps(default_commands, indent=2), height=220)
    try:
        commands = json.loads(raw)
    except json.JSONDecodeError as exc:
        st.error(f"JSON error: {exc}")
        return

    ue_mode = "fake"
    if st.session_state.get("mode") == "live":
        ue_mode = "real"
    if result:
        ue_mode = result.get("ue_mode", ue_mode)
        if result.get("mock") or ue_mode == "fake":
            st.warning("SIMULATED geometry result; the preview and ack are not real Unreal measurements.")

    render_geometry_card(
        commands,
        gold=case.get("gold_commands"),
        expect_clarification=bool(case.get("expect_clarification")),
        case_id=selected_id,
        ue_mode=ue_mode,
    )

    st.subheader("Match readouts")
    try:
        predicted = replay(commands)
        gold = replay(case["gold_commands"])
        difference = max(
            (abs(left[0] - right[0]) + abs(left[1] - right[1]) for left, right in zip(predicted["vertices"], gold["vertices"])),
            default=0.0,
        )
        st.write(
            {
                "case_tolerance": case.get("tolerance"),
                "vertex_count_difference": len(predicted["vertices"]) - len(gold["vertices"]),
                "max_vertex_distance": difference,
            }
        )
    except Exception as exc:
        st.caption(f"Replay not available: {exc}")

    st.subheader("Latency timeline")
    if result_case and result_case.get("latency"):
        stages = {
            key: value
            for key, value in result_case["latency"].items()
            if isinstance(value, (int, float))
        }
        st.plotly_chart(latency_timeline(stages), width="stretch")
        latency_frame = pd.DataFrame([stages])
        st.dataframe(latency_frame, width="stretch")
        from src.ui.components.charts import download_csv

        download_csv(latency_frame, "geometry_latency.csv", "geometry-latency-csv")
    else:
        st.info(NOT_RUN)

    e2e = load_result("e2e_latency")
    if e2e and e2e.get("ue_mode") == "real" and e2e.get("status") == "ok":
        st.subheader("Measured text-to-ack")
        total = e2e.get("latency", {}).get("total", {})
        st.write({"p50": total.get("p50"), "p95": total.get("p95"), "ue_mode": "real"})
    else:
        st.caption("Text-to-ack latency: NOT YET RUN or simulated only.")


if __name__ == "__main__":
    render()
