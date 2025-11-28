import streamlit as st
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
import sys
import librosa

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ==========================================
# Configuration
# ==========================================
st.set_page_config(page_title="Sonic Vision Pro", page_icon="🎵", layout="wide")
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
# Feature Analysis Logic (Tuned for R&B)
# ==========================================
def analyze_track_features(audio_path, genre_probs):
    """
    Analyzes audio features (Energy, Danceability, etc.)
    Combines physical audio features (Librosa) + Genre-based heuristics.
    """
    # 1. Physical Analysis (Librosa)
    y, sr = librosa.load(audio_path, duration=30)
    rms = float(np.mean(librosa.feature.rms(y=y)))  # Loudness/Energy
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)  # Tempo

    # Normalize physical values (Approximation)
    norm_energy = min(rms * 4, 1.0)
    norm_tempo = min((tempo - 60) / 100, 1.0) if tempo > 60 else 0.0

    # 2. Genre-based Weights (Heuristics)
    weights = {
        # Genre:     [Dance, Happy, Acoustic, Instrum]
        'blues': [0.4, 0.3, 0.7, 0.5],
        'classical': [0.1, 0.2, 0.9, 0.9],
        'country': [0.5, 0.6, 0.8, 0.4],
        'disco': [0.9, 0.9, 0.1, 0.2],
        'hiphop': [0.8, 0.5, 0.3, 0.1],
        'jazz': [0.5, 0.4, 0.85, 0.7],
        'metal': [0.2, 0.1, 0.0, 0.1],
        'pop': [0.7, 0.7, 0.45, 0.0],
        'reggae': [0.7, 0.7, 0.4, 0.1],
        'rock': [0.4, 0.3, 0.2, 0.3]
    }

    dance_score = 0.0
    happy_score = 0.0
    acoustic_score = 0.0
    instrum_score = 0.0

    # Weighted sum based on predicted genre probabilities
    for i, prob in enumerate(genre_probs):
        genre = CLASSES[i]
        if genre in weights:
            w = weights[genre]
            dance_score += prob * w[0]
            happy_score += prob * w[1]
            acoustic_score += prob * w[2]
            instrum_score += prob * w[3]

    # 3. Combine Physical + Genre
    final_features = {
        "Energy": int(((norm_energy * 0.4) + ((1 - acoustic_score) * 0.6)) * 100),
        "Danceability": int((norm_tempo * 0.3 + dance_score * 0.7) * 100),
        "Happiness": int(happy_score * 100),
        "Acousticness": int(acoustic_score * 100),
        "Instrumental": int(instrum_score * 100),
        "Loudness (dB)": int(librosa.amplitude_to_db(np.array([rms]))[0])
    }
    return final_features


def plot_circular_bar(ax, value, label, color='#4CAF50'):
    """
    Draws a single circular progress bar (Donut Chart)
    """
    # Data
    if label == "Loudness (dB)":
        display_val = f"{value}dB"
        percentage = min(max((value + 60) / 60, 0), 1)  # Normalize -60dB ~ 0dB
    else:
        display_val = f"{value}"
        percentage = value / 100.0

    # Donut Chart
    sizes = [percentage, 1 - percentage]
    colors = [color, '#f0f2f6']  # Value color, Empty color

    ax.pie(sizes, radius=1, startangle=90, colors=colors,
           wedgeprops=dict(width=0.1, edgecolor='none'), counterclock=False)

    # Center Text
    ax.text(0, 0, display_val, ha='center', va='center', fontsize=20, fontweight='bold', color='#333333')

    # Label Text (Bottom)
    ax.text(0, -1.5, label, ha='center', va='center', fontsize=12, color='#666666')

    ax.axis('equal')


# ==========================================
# UI Layout
# ==========================================
st.title("🎵 Sonic Vision Pro: AI Music Workstation")
st.markdown("""
**The All-in-One AI Tool for Musicians.** Analyze Genre, Visualize Attention, and Deconstruct Stems.
""")

st.sidebar.header("System Info")
st.sidebar.success(f"System Ready (Device: {DEVICE.upper()})")
st.sidebar.info("""
**Features:**
- **Vision:** ResNet18 + Grad-CAM
- **Features:** Spotify-style Analysis
- **Remix:** Demucs (SOTA Separation)
- **Lyrics:** Whisper (Multimodal)
""")

# Load Core AI
with st.spinner("🚀 Booting up Core AI..."):
    cls_model, fe_model, db_vectors, model_path = load_system_resources()

uploaded_file = st.file_uploader("Upload Track (MP3/WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    temp_path = os.path.join("temp_audio.wav")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Tabs
    tab_analyze, tab_remix, tab_lyrics = st.tabs(["📊 Track Features", "🎛️ Remix (Demucs)", "📝 Lyrics (Whisper)"])

    # === TAB 1: Track Features (Spotify Style) ===
    with tab_analyze:
        from predict import process_audio
        from recommend import process_target_song, recommend_songs
        from explain import save_gradcam

        # 1. Genre Prediction First
        inputs = process_audio(temp_path)
        if inputs is not None:
            with torch.no_grad():
                outputs = cls_model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                avg_probs = torch.mean(probs, dim=0).cpu().numpy()

            # 2. Calculate Features
            features = analyze_track_features(temp_path, avg_probs)

            st.subheader("Audio Features Analysis")

            # 3. Draw Circular Bars (Grid Layout)
            fig, axs = plt.subplots(1, 6, figsize=(12, 3))
            cols = st.columns(6)

            # Define colors for each metric
            palette = ['#4CAF50', '#2196F3', '#FFC107', '#9C27B0', '#FF5722', '#607D8B']
            metrics = list(features.items())

            for i, (label, val) in enumerate(metrics):
                plot_circular_bar(axs[i], val, label, color=palette[i])

            st.pyplot(fig)  # Show the matplotlib figure

            st.divider()

            # Genre & Similar Songs
            c1, c2 = st.columns([1, 1])

            with c1:
                st.subheader("Genre Classification")
                top3 = avg_probs.argsort()[-3:][::-1]
                top_genre = CLASSES[top3[0]].upper()
                st.metric("Primary Genre", top_genre, f"{avg_probs[top3[0]] * 100:.1f}% Confidence")
                st.bar_chart({CLASSES[i].upper(): avg_probs[i] for i in top3})

                # Dynamic Genre Insight (Covers ALL Genres)
                genre_insights = {
                    'BLUES': "Identified soulful vocals and guitar licks characteristic of **BLUES**.",
                    'CLASSICAL': "Detected orchestral textures and lack of strong beat patterns (**CLASSICAL**).",
                    'COUNTRY': "AI mapped the **Vocal/Acoustic** texture to 'COUNTRY'.",
                    'DISCO': "Strong dance beat and synth elements identified (**DISCO**).",
                    'HIPHOP': "AI identified strong **Sub-bass & Rhythmic** patterns (**HIPHOP**).",
                    'JAZZ': "Identified as **JAZZ** likely due to harmonic complexity and acoustic elements.",
                    'METAL': "Detected high energy, distorted guitars, and aggressive drumming (**METAL**).",
                    'POP': "Identified as **POP** with modern production features and vocal focus.",
                    'REGGAE': "Distinctive off-beat rhythm and bass patterns detected (**REGGAE**).",
                    'ROCK': "Electric guitar riffs and strong backbeat identified (**ROCK**)."
                }

                # Default message if genre not in dict
                insight_msg = genre_insights.get(top_genre,
                                                 f"AI classified this track as **{top_genre}** based on spectral features.")
                st.info(f"💡 **Insight:** {insight_msg}")

            with c2:
                st.subheader("AI Vision (Attention)")
                if st.button("🔍 View Heatmap"):
                    heatmap_path = "temp_heatmap.jpg"
                    try:
                        save_gradcam(temp_path, model_path, heatmap_path)
                        st.image(heatmap_path, caption="What did AI listen to?", use_container_width=True)
                    except:
                        st.error("Heatmap failed.")

    # === TAB 2: Remix Station ===
    with tab_remix:
        st.header("🎛️ Stem Separation")
        st.info("ℹ️ Using **Meta's Demucs**. Processing takes 3~5 mins on CPU.")

        if st.button("🔥 Start HQ Separation"):
            with st.spinner("Separating... (Please wait)"):
                from separate import separate_audio

                try:
                    stems = separate_audio(temp_path, output_dir="temp_stems")
                    if stems:
                        st.success("Done!")
                        st.session_state['stems'] = stems
                except Exception as e:
                    st.error(f"Error: {e}")

        if 'stems' in st.session_state:
            stems = st.session_state['stems']
            st.markdown("---")
            st.subheader("🎧 Separated Stems")

            c1, c2 = st.columns(2)
            c3, c4 = st.columns(2)
            with c1:
                st.markdown("##### 🎤 Vocals")
                if stems.get('vocals'): st.audio(stems.get('vocals'))
            with c2:
                st.markdown("##### 🥁 Drums")
                if stems.get('drums'): st.audio(stems.get('drums'))
            with c3:
                st.markdown("##### 🎸 Bass")
                if stems.get('bass'): st.audio(stems.get('bass'))
            with c4:
                st.markdown("##### 🎹 Other")
                if stems.get('other'): st.audio(stems.get('other'))

    # === TAB 3: Lyrics ===
    with tab_lyrics:
        st.header("📝 Lyrics Transcription")

        if 'stems' in st.session_state and st.session_state['stems'].get('vocals'):
            target = st.session_state['stems'].get('vocals')
            st.success("✅ Using separated **Vocals** stem.")
        else:
            target = temp_path
            st.warning("⚠️ Using original full track.")

        if st.button("📝 Transcribe (Whisper)"):
            with st.spinner("Listening..."):
                from transcribe import transcribe_audio

                text, lang = transcribe_audio(target)
                if text:
                    st.success(f"Language: {lang}")
                    st.text_area("Lyrics", text, height=300)
                else:
                    st.error("Transcription failed.")