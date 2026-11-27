"""Visible labels for mock, simulated and unmeasured results."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.ui.data import NOT_RUN, is_mock

MOCK_BANNER = "Contains mock results, not measured"


def any_mock(*payloads: Any) -> bool:
    return any(is_mock(payload) for payload in payloads if payload is not None)


def banner(*payloads: Any) -> bool:
    """Page-level banner shown once when any displayed result is mock."""
    if any_mock(*payloads):
        st.warning(f"**{MOCK_BANNER}.** Values are replayed for interface testing only.")
        return True
    return False


def data_status(payload: dict | None) -> str:
    if payload is None:
        return NOT_RUN
    if payload.get("status") == "blocked":
        return NOT_RUN
    if is_mock(payload):
        return "MOCK"
    return "MEASURED"


def with_status(frame: pd.DataFrame, payload: dict | None) -> pd.DataFrame:
    labelled = frame.copy()
    labelled.insert(0, "data", data_status(payload))
    return labelled


def block_label(payload: dict | None, text: str = "not measured") -> None:
    if is_mock(payload):
        st.caption(f":orange-background[**MOCK**] {text}")
    elif payload is not None:
        st.caption(f":green-background[**MEASURED**] {text}")
