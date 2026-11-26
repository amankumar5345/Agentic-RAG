"""The 'How this answer was made' panel."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ui.components.charts import latency_timeline
from src.ui.components.citations import render_citations


def render_trace(trace: dict | None, key_prefix: str) -> None:
    with st.expander("How this answer was made", expanded=False):
        if not trace:
            st.caption("No trace recorded for this answer.")
            return

        guardrail = trace.get("guardrail") or {}
        st.markdown(
            f"**Guardrail:** `{guardrail.get('decision', 'not recorded')}` "
            f"â€” {guardrail.get('category', 'not recorded')}"
        )
        st.markdown(f"**Router role:** `{trace.get('role', 'not recorded')}`")
        st.markdown(
            f"**Retrieval calls:** `{trace.get('retrieval_calls', 'not recorded')}`"
        )

        chunks = trace.get("chunks") or []
        if chunks:
            frame = pd.DataFrame(chunks)
            columns = [column for column in ("id", "doc", "section", "score") if column in frame]
            st.dataframe(frame[columns], width="stretch")
        else:
            st.caption("No retrieved chunks recorded.")

        render_citations(trace.get("citations"), f"{key_prefix}-trace")

        validator = trace.get("citation_validation") or {}
        st.markdown(
            f"**Citation validator:** `{validator.get('result', 'not recorded')}` "
            f"â€” {validator.get('detail', '')}"
        )

        stages = trace.get("stage_latencies") or {}
        numeric = {name: value for name, value in stages.items() if isinstance(value, (int, float))}
        if numeric:
            st.plotly_chart(latency_timeline(numeric), width="stretch")
        else:
            st.caption("No stage latencies recorded.")

        tokens = trace.get("tokens") or {}
        st.markdown(
            f"**Tokens:** `{tokens.get('prompt', 'not recorded')}` prompt / "
            f"`{tokens.get('completion', 'not recorded')}` completion"
        )
        st.markdown(f"**Cost:** `{trace.get('cost_usd', 'not recorded')}` USD")
        if trace.get("mock"):
            st.caption("MOCK trace; not a live result.")

