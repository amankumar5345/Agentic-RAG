"""Chat input, audio transcription, streaming and trace display."""

from __future__ import annotations

import streamlit as st

from src.services.audio_service import transcribe_audio_stream
from src.ui.components.trace_panel import render_trace
from src.ui.state import allow_live_call, log_prompt


def _stream_words(text: str):
    for word in text.split():
        yield word + " "


def _ask(question: str, trace_fn) -> None:
    if not question.strip():
        return
    if not allow_live_call():
        return
    log_prompt(question)
    with st.spinner("Running the agent..."):
        result = trace_fn(question)
    answer = result.get("answer", "I can't support this from the documents.")
    st.write_stream(_stream_words(answer))
    st.session_state.chat_history.append(
        {"role": "user", "content": question, "trace": None}
    )
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "trace": result.get("trace"),
            "geometry": result.get("geometry"),
        }
    )
    render_trace(result.get("trace"), f"chat-{len(st.session_state.chat_history)}")
    if result.get("geometry"):
        from src.ui.components.geometry_card import render_geometry_card

        render_geometry_card(
            result["geometry"],
            ue_mode="fake",
            case_id=f"chat-{len(st.session_state.chat_history)}",
        )


def render_chat(trace_fn) -> None:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("trace"):
                render_trace(message["trace"], f"history-{id(message)}")

    pending = st.session_state.pop("pending_question", None)
    if pending:
        _ask(pending, trace_fn)

    audio = st.audio_input("Record a question")
    if audio is not None and st.button("Transcribe and ask"):
        try:
            question = transcribe_audio_stream(audio)
        except Exception as exc:
            st.error(f"Transcription failed: {exc}")
        else:
            if question:
                _ask(question, trace_fn)

    question = st.chat_input("Ask about lithography, cleanroom safety or maintenance")
    if question:
        _ask(question, trace_fn)
