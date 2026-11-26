"""Methodology: dataset card, protocol, limitations, reproduction and architecture."""

from __future__ import annotations

import streamlit as st

from litho_eval.stats import DEFAULT_RESAMPLES, DEFAULT_SEED
from src.ui.data import load_arm, load_doc, load_result
from src.ui.state import setup_page


def render() -> None:
    setup_page("Methodology")
    st.title("Methodology")

    st.markdown("## Dataset card")
    st.markdown(load_doc("bench/DATASET_CARD.md") or "NOT YET RUN")

    st.markdown("## Protocol")
    comparison = load_result("comparison")
    arm = load_arm("A5") or {}
    models = arm.get("models", {})
    st.markdown(
        f"- Bootstrap resamples: `{DEFAULT_RESAMPLES}`, seed `{DEFAULT_SEED}`.\n"
        f"- Split: dev/test assigned by a fixed seed; QA passages never cross splits.\n"
        f"- Generator: `{models.get('generator', 'not recorded')}`.\n"
        f"- Judge: `{models.get('judge', 'not recorded')}`; judge family must differ "
        "from the generator family.\n"
        f"- Baseline arm: `{comparison.get('baseline') if comparison else 'not recorded'}`."
    )

    st.markdown("## Limitations")
    st.markdown(load_doc("docs/LIMITATIONS.md") or "NOT YET RUN")

    st.markdown("## Reproduce")
    st.markdown(load_doc("docs/REPRODUCE.md") or "NOT YET RUN")

    st.markdown("## Architecture")
    st.markdown(
        """
```mermaid
flowchart LR
    A[User] --> B[Streamlit]
    B --> C[Guardrail]
    C --> D[Router]
    D --> E[Orchestrator]
    E --> F[Retrieval]
    E --> G[Geometry tool]
    F --> H[Generator]
    G --> I[Unreal mailbox]
    H --> J[Citation validator]
    J --> B
    I --> K[Unreal ack]
```
"""
    )


if __name__ == "__main__":
    render()
