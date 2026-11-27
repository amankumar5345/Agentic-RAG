"""The 'How this answer was made' panel."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ui.components.charts import latency_timeline
from src.ui.components.citations import render_citations
from src.ui.components.mock import MOCK_BANNER

NOT_RECORDED = "not recorded"


def _table(rows, columns: list[str]) -> None:
    if isinstance(rows, list) and rows:
        frame = pd.DataFrame(rows)
        keep = [column for column in columns if column in frame.columns]
        st.dataframe(frame[keep], width="stretch")
    else:
        st.caption(f"Not recorded: {', '.join(columns)}")


def render_trace(trace: dict | None, key_prefix: str) -> None:
    with st.expander("How this answer was made", expanded=False):
        if not trace:
            st.caption("No trace recorded for this answer.")
            return

        if trace.get("mock"):
            st.warning(f"**{MOCK_BANNER}.** This trace is a replay.")
        st.caption(
            f"Schema `v{trace.get('schema_version', NOT_RECORDED)}` \u00b7 "
            f"source `{trace.get('source', NOT_RECORDED)}`"
            + (" \u00b7 sample" if trace.get("sample") else "")
        )

        guardrail = trace.get("guardrail") or {}
        verdict = guardrail.get("verdict", guardrail.get("decision", NOT_RECORDED))
        st.markdown(
            f"**Guardrail:** `{verdict}` \u2014 {guardrail.get('category', NOT_RECORDED)}"
        )
        router = trace.get("router") or {}
        st.markdown(
            f"**Router role:** `{router.get('role', trace.get('role', NOT_RECORDED))}`"
        )

        retrieval = trace.get("retrieval") or {}
        if isinstance(retrieval, dict):
            st.markdown(
                f"**Retrieval calls:** `{retrieval.get('calls', NOT_RECORDED)}`"
            )
            st.markdown("**Rerank candidates (25 \u2192 7)**")
            _table(retrieval.get("candidates"), ["id", "doc", "section", "score"])
            st.markdown("**Selected chunks**")
            _table(retrieval.get("selected"), ["id", "doc", "section", "score"])
        else:
            st.caption(f"Retrieval: {retrieval}")

        chunks = trace.get("chunks")
        if isinstance(chunks, list):
            st.markdown("**Retrieved chunk ids**")
            st.write([chunk.get("id", NOT_RECORDED) for chunk in chunks])
        else:
            st.caption(f"Retrieved chunk ids: {chunks}")

        render_citations(trace.get("citations"), f"{key_prefix}-trace")

        validator = trace.get("citation_check") or trace.get("citation_validation") or {}
        st.markdown(
            f"**Citation check:** `{validator.get('result', NOT_RECORDED)}` "
            f"\u2014 coverage `{validator.get('coverage', NOT_RECORDED)}`, "
            f"precision `{validator.get('precision', validator.get('citation_precision', NOT_RECORDED))}`, "
            f"abstained `{validator.get('abstained', NOT_RECORDED)}`"
        )

        stages = trace.get("stage_latencies") or {}
        if isinstance(stages, dict):
            numeric = {
                name: value
                for name, value in stages.items()
                if isinstance(value, (int, float))
            }
            if numeric:
                st.plotly_chart(latency_timeline(numeric), width="stretch")
            else:
                st.caption("No stage latencies recorded.")
        else:
            st.caption(f"Stage latencies: {stages}")

        tokens = trace.get("tokens") or {}
        if isinstance(tokens, dict):
            st.markdown(
                f"**Tokens:** `{tokens.get('prompt', NOT_RECORDED)}` prompt / "
                f"`{tokens.get('completion', NOT_RECORDED)}` completion"
            )
        else:
            st.caption(f"Tokens: {tokens}")
        st.markdown(f"**Cost:** `{trace.get('cost_usd', NOT_RECORDED)}` USD")
