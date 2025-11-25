import streamlit as st
import os
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Import our backend logic
from model import get_model
from recommend import get_feature_extractor, extract_dataset_features, process_target_song, recommend_songs
from explain import save_gradcam
from predict import process_audio
# Import Separation Module
from separate import separate_audio

# ==========================================
# Configuration & Caching
# ==========================================
st.set_page_config(page_title="Sonic Vision", page_icon="🎵", layout="wide")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


@st.cache_resource
def load_system_resources():
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


with st.spinner("🚀 Booting up AI Engine... (Indexing 1,000 songs)"):
    cls_model, fe_model, db_vectors, model_path = load_system_resources()

CLASSES = ['blues', 'classical', 'country', 'disco', 'hiphop',
           'jazz', 'metal', 'pop', 'reggae', 'rock']

# ==========================================
# UI Layout
# ==========================================
st.title("🎵 Sonic Vision: Ultimate AI Music Tool")
st.markdown("""
**Analyze, Visualize, and Deconstruct Music.** Experience the power of Deep Learning: from Classification to **Source Separation (Stem Splitting)**.
""")

st.sidebar.header("About Project")
st.sidebar.info("""
**Developed by PARFUMDEWALKER**
- **Analyze:** Genre Classification
- **Explain:** XAI (Grad-CAM)
- **Deconstruct:** Source Separation (U-Net)
""")

uploaded_file = st.file_uploader("Upload an MP3/WAV file", type=["mp3", "wav"])

if uploaded_file is not None:
    temp_path = os.path.join("temp_audio.wav")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Layout: Tabs for different tasks
    tab_analyze, tab_remix = st.tabs(["📊 Analysis & XAI", "🎛️ Remix Station (Source Separation)"])

    # === TAB 1: Analysis (Existing Features) ===
    with tab_analyze:
        col1, col2 = st.columns([1, 1.5])
        detected_genre = None

        with col2:
            st.subheader("1. Genre Classification")
            with st.spinner("Listening..."):
                inputs = process_audio(temp_path)
                if inputs is not None:
                    with torch.no_grad():
                        outputs = cls_model(inputs)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        avg_probs = torch.mean(probs, dim=0).cpu().numpy()

                    top3_indices = avg_probs.argsort()[-3:][::-1]
                    top_genre = CLASSES[top3_indices[0]].upper()
                    detected_genre = top_genre
                    top_score = avg_probs[top3_indices[0]] * 100

                    st.metric(label="Primary Genre", value=f"{top_genre}", delta=f"{top_score:.1f}% Confidence")
                    chart_data = {CLASSES[i].upper(): avg_probs[i] for i in top3_indices}
                    st.bar_chart(chart_data)

            st.subheader("2. Similar Songs")
            with st.spinner("Searching Database..."):
                target_vec = process_target_song(fe_model, temp_path)
                if target_vec is not None:
                    recs = recommend_songs(target_vec, db_vectors)
                    for name, score in recs:
                        st.write(f"**{name}** (Similarity: `{score * 100:.1f}%`)")
                        st.progress(int(score * 100))

        with col1:
            st.subheader("3. Explainable AI (XAI)")
            st.audio(uploaded_file)

            # Grad-CAM
            with st.expander("🔍 See AI Attention (Grad-CAM)", expanded=True):
                heatmap_path = "temp_heatmap.jpg"
                try:
                    save_gradcam(temp_path, model_path, heatmap_path)
                    st.image(heatmap_path, caption="Red areas = What AI focused on", use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")

    # === TAB 2: Remix Station (New Feature!) ===
    with tab_remix:
        st.header("🎛️ AI Source Separation (U-Net)")
        st.markdown("Isolate Vocals, Drums, Bass, and Other instruments using Deep Learning.")

        st.info("ℹ️ Using Meta's 'htdemucs' model. Highest quality, but takes 3~5 mins on CPU.")
        if st.button("🔥 Start HQ Separation"):
            # [Update] UI Text Updated to reflect CPU usage
            with st.spinner("Splitting Audio Stems... (Processing on CPU for stability)"):
                try:
                    # Run Separation
                    stems = separate_audio(temp_path, output_dir="temp_stems")

                    if not stems:
                        st.error("Separation returned empty results. Check inputs.")
                    else:
                        st.success("Separation Complete! Listen to the stems below.")

                        # Display Players in Grid
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("### 🎤 Vocals")
                            st.audio(stems.get('vocals'))
                            st.markdown("### 🥁 Drums")
                            st.audio(stems.get('drums'))
                        with c2:
                            st.markdown("### 🎸 Bass")
                            st.audio(stems.get('bass'))
                            st.markdown("### 🎹 Other (Inst)")
                            st.audio(stems.get('other'))

                except Exception as e:
                    st.error(f"Separation failed: {e}")