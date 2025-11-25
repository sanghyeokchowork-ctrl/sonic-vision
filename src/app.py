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

# ==========================================
# Configuration & Caching
# ==========================================
st.set_page_config(page_title="Sonic Vision", page_icon="🎵", layout="wide")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# Cache the model and database resources to prevent reloading on every interaction
@st.cache_resource
def load_system_resources():
    # 1. Define Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_path = os.path.join(project_root, 'models', 'best_model.pth')
    data_dir = os.path.join(project_root, 'data', 'processed')

    # 2. Load Classification Model
    cls_model = get_model(num_classes=10, device=DEVICE)
    cls_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    cls_model.eval()

    # 3. Load Feature Extractor & Index Database
    # Note: Feature extraction for 1,000 songs takes time (~40s). Caching is essential.
    fe_model = get_feature_extractor(model_path)
    db_vectors = extract_dataset_features(fe_model, data_dir)

    return cls_model, fe_model, db_vectors, model_path


# Load resources (Runs only once at startup)
with st.spinner("🚀 Booting up AI Engine... (Indexing 1,000 songs from GTZAN)"):
    cls_model, fe_model, db_vectors, model_path = load_system_resources()

CLASSES = ['blues', 'classical', 'country', 'disco', 'hiphop',
           'jazz', 'metal', 'pop', 'reggae', 'rock']

# ==========================================
# UI Layout
# ==========================================
st.title("🎵 Sonic Vision: AI Music Analyzer")
st.markdown("""
**Explainable AI Music Analysis from an R&B Artist's Perspective.** Upload an audio file to visualize **Genre Classification, Similarity Search, and AI Attention Heatmaps.**
""")

st.sidebar.header("About Project")
st.sidebar.info("""
**Developed by PARFUMDEWALKER**
- **Core:** ResNet18 (Transfer Learning)
- **Data:** GTZAN + Custom R&B Tracks
- **Tech:** CNN, Grad-CAM, Vector Search
- **Goal:** Analyzing how AI interprets 'R&B' which is absent in standard datasets.
""")

uploaded_file = st.file_uploader("Upload an MP3/WAV file", type=["mp3", "wav"])

if uploaded_file is not None:
    # Save uploaded file temporarily for processing
    temp_path = os.path.join("temp_audio.wav")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Create a 2-column layout
    col1, col2 = st.columns([1, 1.5])

    # Column 1: Audio Player & XAI Visualization
    with col1:
        st.subheader("🎧 Audio & Heatmap")
        st.audio(uploaded_file)

        with st.spinner("Generating XAI Heatmap..."):
            heatmap_path = "temp_heatmap.jpg"
            try:
                # Generate Grad-CAM heatmap
                save_gradcam(temp_path, model_path, heatmap_path)
                st.image(heatmap_path, caption="Grad-CAM: What did the AI listen to?", use_container_width=True)
            except Exception as e:
                st.error(f"Grad-CAM Error: {e}")

    # Column 2: Analysis Results
    with col2:
        st.subheader("📊 Analysis Results")

        # 1. Genre Classification
        with st.spinner("Analyzing Genre..."):
            inputs = process_audio(temp_path)
            if inputs is not None:
                with torch.no_grad():
                    outputs = cls_model(inputs)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    avg_probs = torch.mean(probs, dim=0).cpu().numpy()

                # Get Top 3 Predictions
                top3_indices = avg_probs.argsort()[-3:][::-1]
                top_genre = CLASSES[top3_indices[0]].upper()
                top_score = avg_probs[top3_indices[0]] * 100

                # Display Primary Genre
                st.metric(label="Primary Genre", value=f"{top_genre}", delta=f"{top_score:.1f}% Confidence")

                # Display Probability Bar Chart
                chart_data = {CLASSES[i].upper(): avg_probs[i] for i in top3_indices}
                st.bar_chart(chart_data)

                # Dynamic Insight Generation
                if top_genre in ['JAZZ', 'POP'] and "jazz" in uploaded_file.name.lower():
                    st.info(
                        f"💡 Insight: Even though the title implies 'Jazz', the AI detected features closer to **{top_genre}** (likely due to modern bass/mixing balance).")
                elif top_genre == 'COUNTRY':
                    st.info(
                        f"💡 Insight: The AI mapped the **Vocal/Acoustic** texture of this track to 'COUNTRY', as R&B is not in the training set.")
                elif top_genre == 'HIPHOP':
                    st.info(
                        f"💡 Insight: The AI successfully identified the **Sub-bass & Rhythmic patterns** characteristic of Hip-Hop.")

        # 2. Similarity Recommendation
        st.divider()
        st.subheader("❤️ Similar Songs (Recommendation)")
        with st.spinner("Searching Database..."):
            # Extract feature vector from the uploaded song
            target_vec = process_target_song(fe_model, temp_path)

            if target_vec is not None:
                # Find top matches using Cosine Similarity
                recs = recommend_songs(target_vec, db_vectors)

                for name, score in recs:
                    st.write(f"**{name}** (Similarity: `{score * 100:.1f}%`)")
                    st.progress(int(score * 100))

    # Optional: Cleanup temp file
    # if os.path.exists(temp_path): os.remove(temp_path)