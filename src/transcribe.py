import whisper
import torch
import os

DEVICE = "cpu"


def transcribe_audio(file_path):
    """
    Transcribes lyrics from audio using OpenAI Whisper.
    """
    filename = os.path.basename(file_path)
    print(f"🗣️ Transcribing Lyrics: {filename}...")

    try:
        # 1. Load Model
        print("⏳ Loading Whisper Model (base)...")
        model = whisper.load_model("base", device=DEVICE)

        # 2. Run Transcription
        result = model.transcribe(file_path, fp16=False)

        text = result["text"].strip()
        language = result["language"]

        print(f"   ✅ Detected Language: {language}")
        return text, language

    except Exception as e:
        print(f"❌ Transcription Failed: {e}")
        return None, None


if __name__ == "__main__":
    # Test Code
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    test_song = os.path.join(project_root, "data", "my_songs", "9624 SERENADE PT. II AR.wav")

    if os.path.exists(test_song):
        text, lang = transcribe_audio(test_song)
        print(f"Lyrics ({lang}): {text[:100]}...")