"""Plotly chart builders with a consistent arm palette."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ARM_COLORS = {
    "A0": "#4C566A",
    "A1": "#0072B2",
    "A2": "#009E73",
    "A3": "#CC79A7",
    "A4": "#D55E00",
    "A5": "#E69F00",
    "A6": "#56B4E9",
    "A7": "#F0E442",
    "A8": "#7B3294",
}


def bar_with_error(
    frame: pd.DataFrame,
    x: str,
    y: str,
    low: str | None,
    high: str | None,
    title: str,
    y_label: str,
):
    colors = [ARM_COLORS.get(str(value), "#666666") for value in frame[x]]
    error_y = None
    if low and high and low in frame and high in frame:
        error_y = dict(
            type="data",
            symmetric=False,
            array=(frame[high] - frame[y]).clip(lower=0),
            arrayminus=(frame[y] - frame[low]).clip(lower=0),
        )
    figure = go.Figure(
        go.Bar(
            x=frame[x],
            y=frame[y],
            marker_color=colors,
            error_y=error_y,
            text=frame[y].round(3),
            textposition="outside",
        )
    )
    figure.update_layout(
        title=title,
        yaxis_title=y_label,
        xaxis_title="Arm",
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return figure


def scatter_cost_latency(frame: pd.DataFrame):
    colors = [ARM_COLORS.get(str(value), "#666666") for value in frame["arm"]]
    figure = go.Figure(
        go.Scatter(
            x=frame["cost"],
            y=frame["p95"],
            mode="markers+text",
            text=frame["arm"],
            textposition="top center",
            marker=dict(size=14, color=colors),
        )
    )
    figure.update_layout(
        title="Cost per correct answer vs p95 latency",
        xaxis_title="Cost per correct answer (USD)",
        yaxis_title="p95 latency (s)",
        template="plotly_white",
    )
    return figure


def confusion_heatmap(matrix: dict, title: str):
    categories = sorted(matrix)
    z = []
    for category in categories:
        row = matrix[category]
        z.append([row.get("tp", 0), row.get("fp", 0), row.get("fn", 0), row.get("tn", 0)])
    figure = go.Figure(
        go.Heatmap(
            z=z,
            x=["TP", "FP", "FN", "TN"],
            y=categories,
            colorscale="Blues",
            text=z,
            texttemplate="%{text}",
        )
    )
    figure.update_layout(title=title, template="plotly_white", height=320)
    return figure


def latency_timeline(stages: dict):
    labels = [name for name, value in stages.items() if isinstance(value, (int, float))]
    values = [stages[name] for name in labels]
    figure = go.Figure(go.Bar(x=labels, y=values, marker_color="#0072B2"))
    figure.update_layout(
        title="Stage latency",
        yaxis_title="seconds",
        xaxis_title="stage",
        template="plotly_white",
    )
    return figure


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def download_csv(frame: pd.DataFrame, filename: str, key: str) -> None:
    st.download_button(
        "Download CSV",
        data=csv_bytes(frame),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def takeaway(text: str) -> None:
    st.caption(f"**Takeaway:** {text}")
