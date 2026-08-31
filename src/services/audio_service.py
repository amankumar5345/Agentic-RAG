import os
import streamlit as st
import speech_recognition as sr

UNREAL_AUDIO_PATH = r"G:\litho\Saved\UserAudioInput.wav.wav"
PROCESSING_AUDIO_PATH = r"G:\litho\Saved\processing_recording.wav"

def external_unreal_audio_exists() -> bool:
    return os.path.exists(UNREAL_AUDIO_PATH)

def transcribe_local_wav_file(audio_path: str = PROCESSING_AUDIO_PATH) -> str | None:
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
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
        if os.path.exists(PROCESSING_AUDIO_PATH):
            os.remove(PROCESSING_AUDIO_PATH) 
    except Exception as e:
        st.error(f"Failed to move/delete the audio file: {e}")

@st.fragment(run_every=2)
def watch_for_unreal_audio():
    if external_unreal_audio_exists():
        st.markdown('***🎙️ Processing Unreal Input...***')
        try:
            if os.path.exists(PROCESSING_AUDIO_PATH):
                os.remove(PROCESSING_AUDIO_PATH)
            os.rename(UNREAL_AUDIO_PATH, PROCESSING_AUDIO_PATH)
        except Exception as e:
            st.write(f"Waiting for file lock to clear... ({e})")
            return

        with st.spinner("Unreal Engine audio detected! Transcribing..."):
            final_prompt = transcribe_local_wav_file()

            if final_prompt: 
                st.session_state.pending_unreal_audio = final_prompt
                st.rerun()
