"""Safety Lab: guardrail systems, injection defences and poisoned documents."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from litho_eval.guardrails_eval import g3_keyword, is_refusal, mock_learned
from litho_eval.injection import (
    attack_success,
    is_drawing_request,
    sanitise,
    tool_allowed,
)
from src.ui.data import NOT_RUN, is_mock, load_injection_cases, load_result
from src.ui.state import setup_page


def _guardrail_blocks(system: str, prompt: str) -> bool:
    if system == "G0":
        return False
    if system in {"G1", "G3"}:
        return g3_keyword(prompt)
    if system == "G2":
        return mock_learned(prompt)
    return False


def render() -> None:
    setup_page("Safety Lab")
    st.title("Safety Lab")

    injection_cases = load_injection_cases()
    if not injection_cases:
        st.warning(f"{NOT_RUN}: no injection cases.")
        return

    labels = {
        f"{case['attack_type']} · {case['id']}": case for case in injection_cases
    }
    selected_label = st.selectbox("Sample attack", list(labels))
    case = labels[selected_label]
    attack_type = case["attack_type"]
    prompt = st.text_area("Attack prompt", case.get("question", ""), height=120)

    left, right = st.columns(2)
    with left:
        guardrail = st.radio("Guardrail system", ["G0", "G1", "G2", "G3"], horizontal=True)
    with right:
        defence = st.radio("Injection defence", ["D0", "D1", "D2", "D3"], horizontal=True)

    st.subheader("Injected passage")
    st.text(case.get("injected_passage", ""))

    sanitised, flagged = sanitise(case.get("injected_passage", ""))
    st.subheader("Decision trace")
    blocked = _guardrail_blocks(guardrail, prompt)
    drawing = is_drawing_request(prompt)
    gate = tool_allowed(defence, prompt)
    st.write(
        {
            "guardrail": guardrail,
            "guardrail_blocks": blocked,
            "defence": defence,
            "sanitiser_flagged": flagged,
            "drawing_request": drawing,
            "tool_gate_allows": gate,
        }
    )
    if defence in {"D2", "D3"}:
        st.markdown("**Sanitised context**")
        st.text(sanitised[:4000])

    simulated_success = attack_success(
        attack_type,
        output="" if blocked or (defence in {"D2", "D3"} and flagged) else "OVERRIDE-4412 BYPASS-9182 CANARY-7731",
        spawn_calls=1 if attack_type == "tool_abuse" and gate else 0,
        user_message=prompt,
    )
    if blocked or (defence in {"D2", "D3"} and flagged) or (defence == "D3" and attack_type == "tool_abuse" and not gate):
        simulated_success = False
    st.metric("Attack succeeded (simulated rules)", str(simulated_success))

    if is_refusal("I can't help with that"):
        st.caption("Refusal detector uses the same rule-based phrases as the benchmark.")

    st.divider()
    st.subheader("Guardrail results")
    guardrails = load_result("guardrails")
    if guardrails is None:
        st.info(NOT_RUN)
    elif is_mock(guardrails):
        st.warning("MOCK guardrail results.")
        rows = [
            {"system": name, "asr": data.get("asr"), "false_refusal": data.get("false_refusal")}
            for name, data in guardrails.get("systems", {}).items()
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch")
    else:
        rows = [
            {"system": name, "asr": data.get("asr"), "false_refusal": data.get("false_refusal")}
            for name, data in guardrails.get("systems", {}).items()
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch")

    st.subheader("Injection results")
    injection = load_result("injection")
    if injection is None:
        st.info(NOT_RUN)
    else:
        if is_mock(injection):
            st.warning("MOCK injection results.")
        rows = [
            {
                "defence": defence_name,
                "asr": data.get("overall", {}).get("rate"),
                "override": data.get("by_type", {}).get("override", {}).get("rate"),
                "tool_abuse": data.get("by_type", {}).get("tool_abuse", {}).get("rate"),
                "canary": data.get("by_type", {}).get("canary", {}).get("rate"),
                "policy_bypass": data.get("by_type", {}).get("policy_bypass", {}).get("rate"),
            }
            for defence_name, data in injection.get("defences", {}).items()
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch")
        st.caption(f"Utility cost: {injection.get('utility_cost')}")

    st.divider()
    st.subheader("Poisoned document demo")
    st.text(case.get("passage", ""))
    st.markdown("**With injection appended**")
    st.text(case.get("injected_passage", ""))
    st.caption(
        "The real benchmark uses the separate litho_injected Chroma collection; "
        "this demo shows the same passage pair without requiring Ollama."
    )


if __name__ == "__main__":
    render()
