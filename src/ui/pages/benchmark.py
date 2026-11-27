"""Benchmark headline page.

Measured results and mock replays are separated. Mock values are shown only
in a labelled replay section; they never feed a headline takeaway, a chart
title without a MOCK prefix, or the README generator.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ui.components.charts import (
    bar_with_error,
    confusion_heatmap,
    download_csv,
    scatter_cost_latency,
    takeaway,
)
from src.ui.components.mock import banner, block_label, with_status
from src.ui.data import NOT_RUN, is_mock, load_arm, load_result, status_label
from src.ui.state import setup_page


def _arms_frame() -> pd.DataFrame:
    rows = []
    for index in range(9):
        arm = f"A{index}"
        payload = load_arm(arm)
        if not payload:
            continue
        if arm == "A8" and (
            payload.get("mock")
            or payload.get("status") == "blocked"
            or not payload.get("aggregates")
        ):
            # A8 is deliberately NOT YET RUN until a real run exists.
            continue
        aggregate = payload.get("aggregates", {})
        correctness = aggregate.get("answer_correctness")
        correctness_ci = aggregate.get("answer_correctness_ci95") or [correctness, correctness]
        correctness_lo = (
            correctness_ci[0]
            if correctness_ci and correctness_ci[0] is not None
            else correctness
        )
        correctness_hi = (
            correctness_ci[1]
            if correctness_ci and len(correctness_ci) > 1 and correctness_ci[1] is not None
            else correctness
        )
        recall = aggregate.get("retrieval", {}).get("recall_at_k")
        recall_ci = aggregate.get("retrieval", {}).get("recall_at_k_ci95") or [recall, recall]
        recall_lo = recall_ci[0] if recall_ci and recall_ci[0] is not None else recall
        recall_hi = (
            recall_ci[1]
            if recall_ci and len(recall_ci) > 1 and recall_ci[1] is not None
            else recall
        )
        latency = aggregate.get("latency_s", {}).get("p95")
        cost = aggregate.get("cost_usd", {}).get("per_correct_answer")
        rows.append(
            {
                "arm": arm,
                "correctness": correctness,
                "correctness_lo": correctness_lo,
                "correctness_hi": correctness_hi,
                "recall": recall,
                "recall_lo": recall_lo,
                "recall_hi": recall_hi,
                "p95": latency,
                "cost": cost,
                "mock": bool(payload.get("mock")),
            }
        )
    return pd.DataFrame(rows)


def _measured_block(frame: pd.DataFrame, comparison: dict | None) -> None:
    if frame.empty:
        return
    st.subheader("Arm comparison (measured)")
    baseline = st.selectbox(
        "Baseline arm",
        sorted(frame["arm"]),
        index=0,
    )
    table = frame.copy()
    if comparison and not comparison.get("mock") and baseline in comparison.get("arms", {}):
        deltas = []
        for arm in table["arm"]:
            metric = (
                comparison["arms"].get(arm, {})
                .get("metrics", {})
                .get("answer_correctness", {})
            )
            deltas.append(metric.get("delta"))
        table["delta_vs_baseline"] = deltas

    st.dataframe(with_status(table, {"mock": False}), width="stretch")
    download_csv(with_status(table, {"mock": False}), "arms.csv", "arms-table")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            bar_with_error(
                table,
                "arm",
                "correctness",
                "correctness_lo",
                "correctness_hi",
                "Answer correctness",
                "correctness",
            ),
            width="stretch",
        )
        download_csv(
            table[["arm", "correctness", "correctness_lo", "correctness_hi"]],
            "correctness.csv",
            "correctness-csv",
        )
    with right:
        st.plotly_chart(
            bar_with_error(
                table,
                "arm",
                "recall",
                "recall_lo",
                "recall_hi",
                "Retrieval recall@k",
                "recall@k",
            ),
            width="stretch",
        )
        download_csv(
            table[["arm", "recall", "recall_lo", "recall_hi"]],
            "recall.csv",
            "recall-csv",
        )

    scored = table.dropna(subset=["correctness"]).sort_values("correctness", ascending=False)
    recalled = table.dropna(subset=["recall"]).sort_values("recall", ascending=False)
    if not scored.empty and not recalled.empty:
        takeaway(
            f"Highest correctness is {scored.iloc[0]['arm']}; "
            f"highest recall@k is {recalled.iloc[0]['arm']}."
        )

    scatter = table.dropna(subset=["cost", "p95"])
    if not scatter.empty:
        st.plotly_chart(scatter_cost_latency(scatter), width="stretch")
        download_csv(scatter[["arm", "cost", "p95"]], "cost_latency.csv", "scatter-csv")
        cheapest = scatter.sort_values("cost").iloc[0]
        fastest = scatter.sort_values("p95").iloc[0]
        takeaway(
            f"Cheapest per correct answer is {cheapest['arm']}; "
            f"lowest p95 is {fastest['arm']}."
        )

    if comparison and not comparison.get("mock"):
        ablation = pd.DataFrame(
            [
                {
                    "arm": arm,
                    "correctness_delta": comparison["arms"][arm]["metrics"]
                    .get("answer_correctness", {})
                    .get("delta"),
                }
                for arm in comparison.get("arms", {})
            ]
        )
        st.subheader("Ablation")
        st.dataframe(ablation, width="stretch")
        download_csv(ablation, "ablation.csv", "ablation-csv")


def _mock_block(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    st.subheader("Mock replay (not measured)")
    st.caption(
        "Mock values are replayed for interface testing only. They are excluded "
        "from headline metrics, comparison deltas, README blocks and claim checks."
    )
    display = frame.copy()
    display["data"] = "MOCK"
    st.dataframe(
        display[
            ["data", "arm", "correctness", "recall", "p95", "cost"]
        ],
        width="stretch",
    )
    download_csv(display, "mock_arms.csv", "mock-arms-csv")
    st.plotly_chart(
        bar_with_error(
            display,
            "arm",
            "correctness",
            "correctness_lo",
            "correctness_hi",
            "MOCK \u2014 answer correctness (not measured)",
            "correctness",
        ),
        width="stretch",
    )


def render() -> None:
    setup_page("Benchmark")
    st.title("Benchmark")
    st.caption("Every number on this page is read from results/*.json.")

    comparison = load_result("comparison")
    frame = _arms_frame()
    guardrails = load_result("guardrails")
    injection = load_result("injection")
    geometry = load_result("geometry")
    citations = load_result("citations")
    judge = load_result("judge_validation")

    banner(
        comparison,
        *[load_arm(f"A{index}") for index in range(9)],
        guardrails,
        injection,
        geometry,
        citations,
        judge,
    )

    if comparison is None and frame.empty and guardrails is None and injection is None:
        st.warning(f"{NOT_RUN}: no arm comparison or per-arm results.")
        return

    if frame.empty:
        st.info(f"{NOT_RUN}: no recorded arm results.")
    else:
        measured = frame[~frame["mock"]]
        mocked = frame[frame["mock"]]
        st.caption(
            "A8 is NOT YET RUN until a real A8 result exists; mock A8 values are "
            "not replayed here."
        )
        _measured_block(measured, comparison)
        _mock_block(mocked)

    st.subheader("Guardrails")
    if guardrails is None:
        st.info(NOT_RUN)
    else:
        rows = [
            {
                "system": name,
                "asr": data.get("asr"),
                "false_refusal": data.get("false_refusal"),
                "n": data.get("n"),
            }
            for name, data in guardrails.get("systems", {}).items()
        ]
        table = with_status(pd.DataFrame(rows), guardrails)
        st.dataframe(table, width="stretch")
        download_csv(table, "guardrails.csv", "guardrails-csv")
        block_label(guardrails)
        if not is_mock(guardrails):
            for name, data in guardrails.get("systems", {}).items():
                with st.expander(f"{name} confusion"):
                    matrix = data.get("confusion", {})
                    st.plotly_chart(
                        confusion_heatmap(matrix, f"{name} confusion"),
                        width="stretch",
                    )
                    frame = (
                        pd.DataFrame(matrix).T.reset_index().rename(columns={"index": "category"})
                        if matrix
                        else pd.DataFrame()
                    )
                    if not frame.empty:
                        download_csv(
                            frame,
                            f"{name}_confusion.csv",
                            f"{name}-confusion-csv",
                        )
            if rows:
                best_asr = min(
                    rows,
                    key=lambda row: row["asr"] if row["asr"] is not None else 1.0,
                )
                best_fr = min(
                    rows,
                    key=lambda row: row["false_refusal"]
                    if row["false_refusal"] is not None
                    else 1.0,
                )
                takeaway(
                    f"Lowest ASR is {best_asr['system']}; "
                    f"lowest false refusal is {best_fr['system']}."
                )

    st.subheader("Indirect injection")
    if injection is None:
        st.info(NOT_RUN)
    else:
        rows = [
            {
                "defence": defence,
                "overall": data.get("overall", {}).get("rate"),
                "override": data.get("by_type", {}).get("override", {}).get("rate"),
                "tool_abuse": data.get("by_type", {}).get("tool_abuse", {}).get("rate"),
                "canary": data.get("by_type", {}).get("canary", {}).get("rate"),
                "policy_bypass": data.get("by_type", {}).get("policy_bypass", {}).get("rate"),
            }
            for defence, data in injection.get("defences", {}).items()
        ]
        table = with_status(pd.DataFrame(rows), injection)
        st.dataframe(table, width="stretch")
        download_csv(table, "injection.csv", "injection-csv")
        block_label(injection)
        st.caption(f"Utility cost: {injection.get('utility_cost')}")
        if rows and not is_mock(injection):
            best = min(
                rows,
                key=lambda row: row["overall"] if row["overall"] is not None else 1.0,
            )
            takeaway(f"Lowest injection ASR is {best['defence']}.")

    st.subheader("Geometry")
    if geometry is None:
        st.info(NOT_RUN)
    else:
        metrics = geometry.get("metrics", {})
        rows = [
            {
                "metric": name,
                "rate": value.get("rate"),
                "ci_low": (value.get("ci95") or [None, None])[0],
                "ci_high": (value.get("ci95") or [None, None])[1],
            }
            for name, value in metrics.items()
            if isinstance(value, dict)
        ]
        table = with_status(pd.DataFrame(rows), geometry)
        st.dataframe(table, width="stretch")
        download_csv(table, "geometry.csv", "geometry-csv")
        block_label(geometry, f"ue_mode: {geometry.get('ue_mode', 'not recorded')}")
        if rows and not is_mock(geometry):
            best = max(
                rows,
                key=lambda row: row["rate"] if row["rate"] is not None else -1.0,
            )
            takeaway(f"Highest geometry metric is {best['metric']}.")

    st.subheader("Citations")
    if citations is None or citations.get("status") != "ok":
        st.info(NOT_RUN)
    else:
        block_label(citations)
        st.json(citations.get("citations"))

    st.subheader("Judge validation")
    if judge is None or judge.get("n", 0) < 50:
        st.info(NOT_RUN)
    else:
        st.write(
            f"Agreement {judge.get('agreement')} \u00b7 kappa {judge.get('kappa')} "
            f"(95% CI {judge.get('kappa_ci95')}) \u00b7 n={judge.get('n')}"
        )

    st.caption(f"Results status: {status_label(comparison)}")


if __name__ == "__main__":
    render()
