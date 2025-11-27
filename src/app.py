import streamlit as st
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ==========================================
# Configuration
# ==========================================
st.set_page_config(page_title="Sonic Vision Pro", page_icon="🎹", layout="wide")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


@st.cache_resource
def load_system_resources():
    # Lazy imports to prevent startup locks
    from model import get_model
    from recommend import get_feature_extractor, extract_dataset_features

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_path = os.path.join(project_root, 'models', 'best_model.pth')
    data_dir = os.path.join(project_root, 'data', 'processed')

    cls_model = get_model(num_classes=10, device=DEVICE)
    cls_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    cls_model.eval()

    fe_model = get_feature_extractor(model_path)
    db_vectors = extract_dataset_features(fe_model, data_dir)

    return cls_model, fe_model, db_vectors, model_path


CLASSES = ['blues', 'classical', 'country', 'disco', 'hiphop',
           'jazz', 'metal', 'pop', 'reggae', 'rock']

# ==========================================
# UI Layout
# ==========================================
st.title("🎹 Sonic Vision Pro: AI Music Workstation")
st.markdown("""
**The All-in-One AI Tool for Musicians.** Analyze Genre, Visualize Attention, Deconstruct Stems, and Convert to MIDI.
""")

st.sidebar.header("System Info")
st.sidebar.success("System Ready (Process Isolation Active)")
st.sidebar.info("""
**Features:**
- **Vision:** ResNet18 + Grad-CAM
- **Remix:** Demucs (SOTA Separation)
- **Lyrics:** Whisper (Multimodal)
- **MIDI:** Basic Pitch (Audio-to-MIDI)
""")

# Load Core AI
with st.spinner("🚀 Booting up Core AI..."):
    cls_model, fe_model, db_vectors, model_path = load_system_resources()

uploaded_file = st.file_uploader("Upload Track (MP3/WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    temp_path = os.path.join("temp_audio.wav")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # [Fix] Define tab variables explicitly
    tab_analyze, tab_remix, tab_multimodal = st.tabs(["📊 Analysis (Vision)", "🎛️ Remix (Demucs)", "📝 Lyrics & MIDI"])

    # === TAB 1: Analysis ===
    with tab_analyze:
        # Lazy Imports
        from predict import process_audio
        from recommend import process_target_song, recommend_songs
        from explain import save_gradcam

        c1, c2 = st.columns([1, 1.5])

        with c2:
            st.subheader("1. Genre Classification")
            inputs = process_audio(temp_path)
            if inputs is not None:
                with torch.no_grad():
                    outputs = cls_model(inputs)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    avg_probs = torch.mean(probs, dim=0).cpu().numpy()

                top3 = avg_probs.argsort()[-3:][::-1]
                top_genre = CLASSES[top3[0]].upper()
                top_score = avg_probs[top3[0]] * 100

                st.metric("Primary Genre", top_genre, f"{top_score:.1f}% Confidence")
                st.bar_chart({CLASSES[i].upper(): avg_probs[i] for i in top3})

                # Insight Logic
                if top_genre in ['JAZZ', 'POP'] and "jazz" in uploaded_file.name.lower():
                    st.info(
                        f"💡 **Insight:** Title implies 'Jazz', but AI detected **{top_genre}** features (Modern Bass/Mixing).")
                elif top_genre == 'COUNTRY':
                    st.info(f"💡 **Insight:** AI mapped the **Vocal/Acoustic** texture to 'COUNTRY'.")
                elif top_genre == 'HIPHOP':
                    st.info(f"💡 **Insight:** AI identified strong **Sub-bass & Rhythmic** patterns.")

            st.divider()
            st.subheader("2. Similar Songs")
            target_vec = process_target_song(fe_model, temp_path)
            if target_vec is not None:
                recs = recommend_songs(target_vec, db_vectors, top_k=3)
                for name, score in recs:
                    st.write(f"**{name}** (`{score * 100:.1f}%`)")
                    st.progress(int(score * 100))

        with c1:
            st.subheader("3. AI Vision (XAI)")
            st.audio(uploaded_file)

            heatmap_path = "temp_heatmap.jpg"
            if st.button("🔍 Generate Heatmap"):
                with st.spinner("Analyzing spectrogram..."):
                    try:
                        save_gradcam(temp_path, model_path, heatmap_path)
                        st.image(heatmap_path, caption="AI Attention", width=400)
                    except:
                        st.error("Heatmap failed.")

    # === TAB 2: Remix Station ===
    with tab_remix:
        st.header("🎛️ Stem Separation")
        st.info("ℹ️ Using **Meta's Demucs (htdemucs)**. Processing takes 3~5 mins on CPU.")

        if st.button("🔥 Start HQ Separation"):
            with st.spinner("Separating Vocals, Drums, Bass, Other... (Please wait)"):
                from separate import separate_audio

                try:
                    stems = separate_audio(temp_path, output_dir="temp_stems")
                    if stems:
                        st.success("Separation Complete!")
                        st.session_state['stems'] = stems

                        # 4-Column Grid
                        c1, c2 = st.columns(2)
                        c3, c4 = st.columns(2)
                        with c1:
                            st.markdown("##### 🎤 Vocals")
                            st.audio(stems.get('vocals'))
                        with c2:
                            st.markdown("##### 🥁 Drums")
                            st.audio(stems.get('drums'))
                        with c3:
                            st.markdown("##### 🎸 Bass")
                            st.audio(stems.get('bass'))
                        with c4:
                            st.markdown("##### 🎹 Other")
                            st.audio(stems.get('other'))
                except Exception as e:
                    st.error(f"Separation Error: {e}")

    # === TAB 3: Multimodal ===
    with tab_multimodal:
        st.header("📝 Lyrics & MIDI")
        c_lyrics, c_midi = st.columns(2)

        with c_lyrics:
            st.subheader("🗣️ Lyrics")
            if st.button("📝 Transcribe (Whisper)"):
                with st.spinner("Transcribing..."):
                    from transcribe import transcribe_audio

                    target = st.session_state.get('stems', {}).get('vocals', temp_path)
                    text, lang = transcribe_audio(target)
                    if text:
                        st.success(f"Language: {lang}")
                        st.text_area("Lyrics", text, height=250)

        with c_midi:
            st.subheader("🎹 MIDI Converter")
            # Use Bass stem if available
            target_for_midi = st.session_state.get('stems', {}).get('bass', temp_path)

            if 'stems' in st.session_state:
                st.success("✅ Using separated **Bass** stem.")
            else:
                st.caption("⚠️ Using original track.")

            if st.button("🎼 Convert to MIDI"):
                with st.spinner("Converting... (Running Basic Pitch in Subprocess)"):
                    # [CRITICAL FIX] Run MIDI conversion in a SEPARATE PROCESS
                    output_dir = "temp_midi"

                    cmd = [sys.executable, "src/midify.py", target_for_midi, output_dir]

                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

                        midi_file = os.path.join(output_dir, "converted.mid")

                        if os.path.exists(midi_file):
                            st.success("Conversion Complete!")
                            with open(midi_file, "rb") as f:
                                st.download_button("⬇️ Download MIDI File", f, file_name="converted.mid")
                        else:
                            st.error("Conversion failed.")
                            st.text("Error Log:")
                            st.code(result.stderr)

                    except Exception as e:
                        st.error(f"Subprocess Error: {e}")