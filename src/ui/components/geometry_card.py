"""Geometry preview, validation, send and ack card."""

from __future__ import annotations

import math
import os
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from src.core.citations import RetrievedChunk  # noqa: F401  (public type nearby)
from src.core.geometry import build_packet, replay, validate_commands, write_packet
from src.services.ue_ack import wait_for_ack
from tools.fake_unreal import process_once


def _preview(commands, gold, tolerance):
    figure = go.Figure()
    try:
        predicted = replay(commands)
        vertices = predicted["vertices"]
        if len(vertices) > 1:
            figure.add_trace(
                go.Scatter(
                    x=[point[0] for point in vertices],
                    y=[point[1] for point in vertices],
                    mode="lines+markers",
                    name="prediction",
                    line=dict(color="#0072B2", width=3),
                )
            )
        for circle in predicted["circles"]:
            figure.add_trace(
                go.Scatter(
                    x=[circle["center"][0]],
                    y=[circle["center"][1]],
                    mode="markers",
                    marker=dict(size=circle["radius"] * 2, color="#0072B2", opacity=0.4),
                    name="circle",
                )
            )
    except Exception as exc:
        figure.add_annotation(text=str(exc), showarrow=False)

    if gold:
        try:
            expected = replay(gold)
            vertices = expected["vertices"]
            if len(vertices) > 1:
                figure.add_trace(
                    go.Scatter(
                        x=[point[0] for point in vertices],
                        y=[point[1] for point in vertices],
                        mode="lines",
                        name="gold",
                        line=dict(color="#D55E00", width=2, dash="dash"),
                    )
                )
        except Exception:
            pass

    figure.update_layout(
        title="2D preview",
        template="plotly_white",
        height=420,
        xaxis_title="x",
        yaxis_title="y",
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    return figure


def render_geometry_card(
    commands,
    gold=None,
    expect_clarification: bool = False,
    case_id: str = "",
    ue_mode: str = "fake",
) -> None:
    st.subheader("Geometry")
    st.json(commands)

    errors = []
    valid = True
    try:
        validate_commands(commands)
    except Exception as exc:
        valid = False
        errors.append(str(exc))
    if expect_clarification:
        st.warning("This case expects a clarification, not a drawing.")
    if errors:
        st.error("\n".join(errors))

    if gold:
        st.caption("Gold commands shown as a dashed outline.")

    try:
        st.plotly_chart(
            _preview(commands, gold, 1.0),
            width="stretch",
            key=f"geo-preview-{case_id}",
        )
    except Exception as exc:
        st.error(f"Preview failed: {exc}")

    if valid and not expect_clarification and st.button(
        "Send to Unreal", key=f"send-{case_id}"
    ):
        mailbox = Path(os.environ.get("LITHO_MAILBOX_DIR", "./runtime/unreal_mailbox"))
        packet = build_packet(commands)
        try:
            write_packet(packet, mailbox)
            if ue_mode == "fake":
                process_once(mailbox, delay_s=0.05)
            with st.spinner("Waiting for Unreal acknowledgement..."):
                ack = wait_for_ack(
                    packet["command_id"],
                    timeout_s=3.0,
                    mailbox_dir=mailbox,
                    poll_interval_s=0.05,
                )
            if ack:
                st.success(f"Ack received: {ack.get('rendered_at')}")
            else:
                st.warning("No ack received; showing a simulated ack.")
                st.json(
                    {
                        "command_id": packet["command_id"],
                        "received_at": "simulated",
                        "rendered_at": "simulated",
                    }
                )
            if ue_mode == "fake":
                st.caption("SIMULATED ack via tools/fake_unreal.py")
        except Exception as exc:
            st.error(f"Send failed: {exc}")

    if gold and valid:
        from src.core.geometry import replay

        try:
            predicted = replay(commands)
            expected = replay(gold)
            distances = [
                math.dist(left, right)
                for left, right in zip(predicted["vertices"], expected["vertices"])
            ]
            difference = max(distances, default=0.0)
            st.metric("Max replay difference", f"{difference:.6g}")
        except Exception:
            st.caption("Exact/geometric match not available for this pair.")

