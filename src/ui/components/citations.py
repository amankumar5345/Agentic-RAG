"""Citation chips and chunk viewer."""

from __future__ import annotations

import streamlit as st


def render_citations(citations: list[dict] | None, key_prefix: str) -> None:
    citations = citations or []
    if not citations:
        st.caption("No citations recorded for this answer.")
        return
    columns = st.columns(min(len(citations), 4))
    for index, citation in enumerate(citations):
        label = f"[{citation.get('id', '?')}]"
        if columns[index % len(columns)].button(
            label,
            key=f"{key_prefix}-cite-{index}",
            help=f"{citation.get('doc', '')} / {citation.get('section', '')}",
        ):
            st.session_state[f"{key_prefix}-open-citation"] = citation.get("id")
    opened = st.session_state.get(f"{key_prefix}-open-citation")
    if opened:
        for citation in citations:
            if citation.get("id") == opened:
                st.markdown(
                    f"**{opened}** — `{citation.get('doc')}` / `{citation.get('section')}`"
                )
                st.info(citation.get("snippet", "No snippet recorded."))
