import io
import re
from gtts import gTTS

def generate_tts_audio(text: str) -> io.BytesIO:
    """
    Generates an in-memory audio byte stream using gTTS.
    Automatically detects if the text is Hindi or English, and cleans HTML tags.
    """
    # Strip HTML tags like <span> or <br> before synthesizing speech
    clean_text = re.sub(r'<[^>]+>', ' ', text).strip()
    if not clean_text:
        clean_text = "No content to speak."
        
    sound_file = io.BytesIO()
    is_hindi = any("\u0900" <= char <= "\u097f" for char in clean_text)
    lang_code = "hi" if is_hindi else "en"
    
    tts = gTTS(clean_text, lang=lang_code)
    tts.write_to_fp(sound_file)
    sound_file.seek(0)
    return sound_file
