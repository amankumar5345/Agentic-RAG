"""Benchmark headline page."""

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
from src.ui.data import (
    NOT_RUN,
    is_mock,
    load_arm,
    load_result,
    status_label,
)
from src.ui.state import setup_page


def _arms_frame() -> pd.DataFrame:
    rows = []
    for index in range(9):
        arm = f"A{index}"
        payload = load_arm(arm)
        if not payload:
            continue
        aggregate = payload.get("aggregates", {})
        correctness = aggregate.get("answer_correctness")
        correctness_ci = aggregate.get("answer_correctness_ci95") or [correctness, correctness]
        correctness_lo = correctness_ci[0] if correctness_ci and correctness_ci[0] is not None else correctness
        correctness_hi = correctness_ci[1] if correctness_ci and len(correctness_ci) > 1 and correctness_ci[1] is not None else correctness
        recall = aggregate.get("retrieval", {}).get("recall_at_k")
        recall_ci = aggregate.get("retrieval", {}).get("recall_at_k_ci95") or [recall, recall]
        recall_lo = recall_ci[0] if recall_ci and recall_ci[0] is not None else recall
        recall_hi = recall_ci[1] if recall_ci and len(recall_ci) > 1 and recall_ci[1] is not None else recall
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


def render() -> None:
    setup_page("Benchmark")
    st.title("Benchmark")
    st.caption("Every number on this page is read from results/*.json.")

    comparison = load_result("comparison")
    frame = _arms_frame()

    if comparison is None and frame.empty:
        st.warning(f"{NOT_RUN}: no arm comparison or per-arm results.")
        return

    if not frame.empty and frame["mock"].any():
        st.warning("MOCK arm results. These are not benchmark numbers.")

    if not frame.empty:
        st.subheader("Arm comparison")
        baseline = st.selectbox(
            "Baseline arm",
            sorted(comparison.get("arms", {})) if comparison else sorted(frame["arm"]),
            index=0,
        )
        table = frame.copy()
        if comparison and baseline in comparison.get("arms", {}):
            deltas = []
            for arm in table["arm"]:
                metric = (
                    comparison["arms"].get(arm, {})
                    .get("metrics", {})
                    .get("answer_correctness", {})
                )
                deltas.append(metric.get("delta"))
            table["delta_vs_baseline"] = deltas
        st.dataframe(table, width="stretch")
        download_csv(table, "arms.csv", "arms-table")

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
            download_csv(table[["arm", "correctness", "correctness_lo", "correctness_hi"]], "correctness.csv", "correctness-csv")
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
            download_csv(table[["arm", "recall", "recall_lo", "recall_hi"]], "recall.csv", "recall-csv")

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

        if comparison:
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
            best = table.dropna(subset=["correctness"]).sort_values("correctness", ascending=False)
            if not best.empty:
                takeaway(f"Highest correctness is {best.iloc[0]['arm']}.")

    guardrails = load_result("guardrails")
    st.subheader("Guardrails")
    if guardrails is None:
        st.info(NOT_RUN)
    else:
        if is_mock(guardrails):
            st.warning("MOCK guardrail results.")
        rows = [
            {
                "system": name,
                "asr": data.get("asr"),
                "false_refusal": data.get("false_refusal"),
                "n": data.get("n"),
            }
            for name, data in guardrails.get("systems", {}).items()
        ]
        table = pd.DataFrame(rows)
        st.dataframe(table, width="stretch")
        download_csv(table, "guardrails.csv", "guardrails-csv")
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
                    download_csv(frame, f"{name}_confusion.csv", f"{name}-confusion-csv")
        if rows:
            best_asr = min(rows, key=lambda row: row["asr"] if row["asr"] is not None else 1.0)
            best_fr = min(
                rows,
                key=lambda row: row["false_refusal"]
                if row["false_refusal"] is not None
                else 1.0,
            )
            takeaway(
                f"Lowest ASR is {best_asr['system']}; lowest false refusal is {best_fr['system']}."
            )

    injection = load_result("injection")
    st.subheader("Indirect injection")
    if injection is None:
        st.info(NOT_RUN)
    else:
        if is_mock(injection):
            st.warning("MOCK injection results.")
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
        table = pd.DataFrame(rows)
        st.dataframe(table, width="stretch")
        download_csv(table, "injection.csv", "injection-csv")
        st.caption(f"Utility cost: {injection.get('utility_cost')}")
        if rows:
            best = min(
                rows,
                key=lambda row: row["overall"] if row["overall"] is not None else 1.0,
            )
            takeaway(f"Lowest injection ASR is {best['defence']}.")

    geometry = load_result("geometry")
    st.subheader("Geometry")
    if geometry is None:
        st.info(NOT_RUN)
    elif geometry.get("mock") or geometry.get("status") != "ok":
        st.info("NOT YET RUN (mock or blocked geometry result)")
    else:
        rows = [
            {
                "category": name,
                "rate": value.get("rate"),
                "ci_low": (value.get("ci95") or [None, None])[0],
                "ci_high": (value.get("ci95") or [None, None])[1],
            }
            for name, value in geometry.get("by_category", {}).items()
        ]
        table = pd.DataFrame(rows)
        st.dataframe(table, width="stretch")
        download_csv(table, "geometry.csv", "geometry-csv")
        if rows:
            best = max(
                rows,
                key=lambda row: row["rate"] if row["rate"] is not None else -1.0,
            )
            takeaway(f"Highest geometry accuracy is {best['category']}.")

    citations = load_result("citations")
    st.subheader("Citations")
    if citations is None or citations.get("mock") or citations.get("status") != "ok":
        st.info(NOT_RUN)
    else:
        st.json(citations.get("citations"))

    judge = load_result("judge_validation")
    st.subheader("Judge validation")
    if judge is None or judge.get("n", 0) < 50:
        st.info(NOT_RUN)
    else:
        st.write(
            f"Agreement {judge.get('agreement')} Â· kappa {judge.get('kappa')} "
            f"(95% CI {judge.get('kappa_ci95')}) Â· n={judge.get('n')}"
        )

    st.caption(f"Results status: {status_label(comparison)}")


if __name__ == "__main__":
    render()

