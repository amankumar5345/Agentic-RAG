
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from src.core.graph import chain
from src.services.translation import translate_to_hindi
from src.services.audio_service import watch_for_unreal_audio, transcribe_audio_stream
from src.services.tts_service import generate_tts_audio

#----------------- page configuration
st.set_page_config(page_title="Project Litho Agentic RAG", page_icon="⚙️", layout="centered")

#----------------- session state initialization 
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'roles' not in st.session_state:
    st.session_state.roles = []  

if 'translated_text' not in st.session_state:
    st.session_state.translated_text = ""

if 'translate_button' not in st.session_state:
    st.session_state.translate_button = False

if 'to_speech_content' not in st.session_state:
    st.session_state.to_speech_content = ""


def load_all_messages():
    """Renders all conversation history in the UI."""
   
    ai_messages = [(m, r) for m, r in zip(st.session_state.messages, st.session_state.roles)
                   if isinstance(m, AIMessage)]
    ai_iter = iter(ai_messages)

    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                try:
                    _, role = next(ai_iter)
                except StopIteration:
                    role = "Assistant"
                display_html = f'<span style="font-size: 22px; font-weight: bold;">{role} :</span><br>{message.content}'
                st.markdown(display_html, unsafe_allow_html=True)


load_all_messages()


def get_response(user_input: str):
    """Executes retrieval and LangGraph multi-agent chain."""
    if not user_input or not user_input.strip():
        return

    st.session_state.translated_text = ""
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("Searching docs..."):

        final_messages = st.session_state.messages[-10:]

        with st.spinner("Generating answer..."):
            final_state = chain.invoke(
                {
                    "messages": final_messages,
                    "rag_calls": 0,
                    "is_blocked": False,
                    "eval_results": None,
                    "role": "",
                }
            )
            role_name = final_state.get('role', 'Assistant')
            message_content = final_state["messages"][-1].content  
            display_html = f'<span style="font-size: 22px; font-weight: bold;">{role_name} :</span><br>{message_content}'

            
            st.session_state.messages.append(AIMessage(content=message_content))
            st.session_state.roles.append(role_name)  
            st.session_state.to_speech_content = message_content
            with st.chat_message("assistant"):
                st.markdown(display_html, unsafe_allow_html=True)

            # Display DeepEval scores if available
            eval_scores = final_state.get('eval_results')
            if eval_scores:
                with st.expander("DeepEval Scores", expanded=False):
                    for metric_name, data in eval_scores.items():
                        if "error" in data:
                            st.error(f"**{metric_name}**: {data['error']}")
                        else:
                            score = data.get('score', 0)
                            passed = data.get('passed', False)
                            st.markdown(f"**{metric_name}**: `{score}` {' (Passed)' if passed else ' (Failed)'}")


#----------------- Unreal Engine Audio Listener 
watch_for_unreal_audio()

if "pending_unreal_audio" in st.session_state:
    audio_text = st.session_state.pending_unreal_audio
    del st.session_state.pending_unreal_audio
    get_response(audio_text)

#--------------------------- Main UI Layout ---------
st.sidebar.title("Project Litho Agentic RAG")
st.sidebar.caption("Guarded Multi-Agent Photolithography & Cleanroom Operations System")

if st.sidebar.button('Clear Chat'):
    st.session_state.messages = []
    st.session_state.roles = []
    st.session_state.translated_text = ""
    st.session_state.to_speech_content = ""
    st.rerun()

if st.session_state.translated_text:
    with st.chat_message("assistant"):
        st.markdown(st.session_state.translated_text)

# Chat and Mic Input
if user_input := st.chat_input("Enter your question:", accept_audio=True):
    if user_input.audio:
        with st.spinner("Transcribing audio..."):
            final_prompt = transcribe_audio_stream(user_input.audio)
        if final_prompt:
            get_response(final_prompt)
    else:
        get_response(user_input.text)

# ----------------Action Buttons----------Translation & Text-to-Speech--------
if len(st.session_state.messages) > 0:
    col1, col2 = st.columns(2)

    with col1:
        if st.button('Translate to Hindi'):
            with st.spinner("Translating..."):
                last_msg = st.session_state.messages[-1].content
                st.session_state.translated_text = translate_to_hindi(last_msg)
                st.session_state.to_speech_content = st.session_state.translated_text
                st.rerun()

    with col2:
        if st.button("Play Audio"):
            with st.spinner("Generating audio..."):
                sound_file = generate_tts_audio(st.session_state.to_speech_content)
                st.audio(sound_file, autoplay=False)
