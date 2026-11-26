import os
from pathlib import Path

import speech_recognition as sr
import streamlit as st


def _unreal_audio_path() -> Path:
    """WAV file written by the Unreal mic manager."""
    return Path(
        os.environ.get("LITHO_AUDIO_PATH", "./runtime/UserAudioInput.wav")
    ).expanduser()


def _processing_audio_path() -> Path:
    """Local copy used while transcribing; kept beside the Unreal input."""
    return _unreal_audio_path().with_name("processing_recording.wav")


def external_unreal_audio_exists() -> bool:
    return _unreal_audio_path().exists()


def transcribe_local_wav_file(audio_path: str | os.PathLike | None = None) -> str | None:
    path = Path(audio_path) if audio_path is not None else _processing_audio_path()
    r = sr.Recognizer()
    try:
        with sr.AudioFile(str(path)) as source:
            audio_data = r.record(source)
            transcription = r.recognize_google(audio_data)
            return transcription
    except sr.UnknownValueError:
        st.error("Could not understand audio from Unreal Engine.")
        return None
    except sr.RequestError as e:
        st.error(f"Google Speech API request failed: {e}")
        return None
    except Exception as e:
        st.error(f"Error reading the audio file: {e}")
        return None

def transcribe_audio_stream(audio_stream) -> str | None:
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_stream) as source:
            audio_data = r.record(source)
            return r.recognize_google(audio_data)
    except sr.UnknownValueError:
        st.error("Could not understand audio.")
        return None
    except sr.RequestError as e:
        st.error(f"Google Speech API request failed: {e}")
        return None
    except Exception as e:
        st.error(f"Error reading audio stream: {e}")
        return None

def clear_or_delete_processed_wav():
    try:
        path = _processing_audio_path()
        if path.exists():
            path.unlink()
    except Exception as e:
        st.error(f"Failed to move/delete the audio file: {e}")

@st.fragment(run_every=2)
def watch_for_unreal_audio():
    if external_unreal_audio_exists():
        st.markdown('***🎙️ Processing Unreal Input...***')
        try:
            unreal_path = _unreal_audio_path()
            processing_path = _processing_audio_path()
            if processing_path.exists():
                processing_path.unlink()
            unreal_path.replace(processing_path)
        except Exception as e:
            st.write(f"Waiting for file lock to clear... ({e})")
            return

        with st.spinner("Unreal Engine audio detected! Transcribing..."):
            final_prompt = transcribe_local_wav_file()

            if final_prompt: 
                st.session_state.pending_unreal_audio = final_prompt
                st.rerun()
