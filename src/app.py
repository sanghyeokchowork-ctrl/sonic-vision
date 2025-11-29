import streamlit as st
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
import sys
import librosa
import pandas as pd

# 사용자 모듈 임포트
from model import get_model
from mixing_assistant import MixingEngineer
from vocal_timbre_model import VocalTimbreCNN
from recommend import load_siamese_model, build_database_index, find_similar_songs

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ==========================================
# Configuration
# ==========================================
st.set_page_config(page_title="Sonic Vision Pro", page_icon="🎵", layout="wide")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

CLASSES = ['blues', 'classical', 'country', 'disco', 'hiphop',
           'jazz', 'metal', 'pop', 'reggae', 'rock']

TIMBRE_TAGS = ['Bright', 'Warm', 'Breathy', 'Rough', 'Clean']


@st.cache_resource
def load_core_models():
    """모든 AI 모델 및 데이터베이스 로드"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # 1. Genre Model
    genre_path = os.path.join(project_root, 'models', 'best_model.pth')
    cls_model = get_model(num_classes=10, device=DEVICE)
    if os.path.exists(genre_path):
        cls_model.load_state_dict(torch.load(genre_path, map_location=DEVICE))
    cls_model.eval()

    # 2. Mixing Engineer (Rule-based)
    mix_engineer = MixingEngineer(sample_rate=22050)

    # 3. Vocal Timbre Model
    timbre_path = os.path.join(project_root, 'models', 'vocal_timbre.pth')
    timbre_model = VocalTimbreCNN(num_tags=len(TIMBRE_TAGS)).to(DEVICE)
    if os.path.exists(timbre_path):
        timbre_model.load_state_dict(torch.load(timbre_path, map_location=DEVICE))
    else:
        print("⚠️ Vocal Timbre model not found.")
    timbre_model.eval()

    # 4. Siamese Network & DB Index
    siamese_path = os.path.join(project_root, 'models', 'siamese_net.pth')
    siamese_model = load_siamese_model(siamese_path)

    # DB Indexing (GTZAN Processed Data)
    processed_dir = os.path.join(project_root, 'data', 'processed')
    if os.path.exists(processed_dir):
        db_vectors = build_database_index(siamese_model, processed_dir)
    else:
        db_vectors = {}
        print("⚠️ 'data/processed' not found. Run preprocess.py first.")

    # [수정됨] model_path 대신 genre_path를 리턴합니다.
    return cls_model, mix_engineer, timbre_model, siamese_model, db_vectors, genre_path, project_root


def analyze_timbre(model, audio_path):
    """보컬 파일의 음색을 추론"""
    try:
        # 학습 때와 동일한 전처리 (3초, 22050Hz, MFCC 40)
        y, sr = librosa.load(audio_path, sr=22050, duration=3.0)
        target_len = 22050 * 3

        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        input_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            # Logits -> Sigmoid (확률값 0~1 변환)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

        return dict(zip(TIMBRE_TAGS, probs))
    except Exception as e:
        return {"Error": str(e)}


# ==========================================
# Main UI
# ==========================================
st.title("🎵 Sonic Vision Pro: AI Music Workstation")
st.markdown("""
**All-in-One AI Tool for Musicians.** Genre Analysis, Stem Separation, Mixing Advice, and Vocal Analysis.
""")

st.sidebar.header("System Status")
st.sidebar.success(f"Device: {DEVICE.upper()}")

# Load Models
with st.spinner("🚀 Booting up AI Engines..."):
    # [수정됨] 받는 쪽 변수명도 model_path로 유지 (내부적으로 genre_path 값을 받음)
    cls_model, mix_engineer, timbre_model, siamese_model, db_vectors, model_path, project_root = load_core_models()

uploaded_file = st.file_uploader("Upload Track (MP3/WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Session State 초기화
    if 'genre_probs' not in st.session_state:
        st.session_state['genre_probs'] = None
    if 'top_genre' not in st.session_state:
        st.session_state['top_genre'] = 'pop'  # default

    # 탭 구성 (5개)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Analysis",
        "🎛️ Remix (Stems)",
        "🎚️ Mixing Studio",
        "🎤 Vocal Lab",
        "🔭 Discovery"
    ])

    # === TAB 1: Analysis ===
    with tab1:
        st.header("📊 Track Analysis")
        from predict import process_audio
        from explain import save_gradcam

        if st.button("🔍 Analyze Genre & Features"):
            inputs = process_audio(temp_path)
            if inputs is not None:
                with torch.no_grad():
                    outputs = cls_model(inputs)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    avg_probs = torch.mean(probs, dim=0).cpu().numpy()

                st.session_state['genre_probs'] = avg_probs
                top3 = avg_probs.argsort()[-3:][::-1]
                st.session_state['top_genre'] = CLASSES[top3[0]]

                c1, c2 = st.columns([1, 1])
                with c1:
                    st.subheader("Genre Prediction")
                    st.metric("Primary Genre", st.session_state['top_genre'].upper(),
                              f"{avg_probs[top3[0]] * 100:.1f}%")
                    st.bar_chart({CLASSES[i]: avg_probs[i] for i in top3})

                with c2:
                    st.subheader("AI Vision (Grad-CAM)")
                    heatmap_path = "temp_heatmap.jpg"
                    try:
                        save_gradcam(temp_path, model_path, heatmap_path)
                        st.image(heatmap_path, caption="AI Attention Map", use_container_width=True)
                    except:
                        st.warning("Heatmap generation failed (Model might be untrained).")

    # === TAB 2: Remix ===
    with tab2:
        st.header("🎛️ Stem Separation (Demucs)")
        if st.button("🔥 Separate Stems"):
            with st.spinner("Separating... (This takes time)"):
                from separate import separate_audio

                stems = separate_audio(temp_path, output_dir="temp_stems")
                if stems:
                    st.session_state['stems'] = stems
                    st.success("Separation Complete!")

        if 'stems' in st.session_state:
            stems = st.session_state['stems']
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown("**🎤 Vocals**")
                if stems.get('vocals'): st.audio(stems.get('vocals'))
            with c2:
                st.markdown("**🥁 Drums**")
                if stems.get('drums'): st.audio(stems.get('drums'))
            with c3:
                st.markdown("**🎸 Bass**")
                if stems.get('bass'): st.audio(stems.get('bass'))
            with c4:
                st.markdown("**🎹 Other**")
                if stems.get('other'): st.audio(stems.get('other'))

    # === TAB 3: Mixing Studio ===
    with tab3:
        st.header("🎚️ AI Mixing Assistant")
        st.info("Analyzes frequency balance and suggests EQ settings based on the target genre.")

        target_genre = st.selectbox(
            "Target Genre Style",
            CLASSES,
            index=CLASSES.index(st.session_state['top_genre']) if st.session_state['top_genre'] in CLASSES else 7
        )

        if st.button("🎚️ Analyze Mix Balance"):
            with st.spinner("Analyzing Frequency Spectrum..."):
                mix_result = mix_engineer.get_mixing_suggestions(temp_path, detected_genre=target_genre)

                if "error" in mix_result:
                    st.error(f"Analysis Failed: {mix_result['error']}")
                else:
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.subheader("💡 Suggestions")
                        if mix_result['alert_level'] == "Red":
                            st.error("Major adjustments needed!")
                        elif mix_result['alert_level'] == "Yellow":
                            st.warning("Adjustments recommended.")
                        else:
                            st.success("Mix is well balanced!")

                        for tip in mix_result['suggestions']:
                            if "Boost" in tip:
                                st.markdown(f"- 🔼 {tip}")
                            elif "Cut" in tip:
                                st.markdown(f"- 🔽 {tip}")
                            else:
                                st.markdown(f"- ✅ {tip}")
                        st.markdown("---")
                        st.caption(mix_result['dynamic_advice'])

                    with col2:
                        st.subheader("Frequency Balance")
                        st.bar_chart(mix_result['balance_data'])

    # === TAB 4: Vocal Lab ===
    with tab4:
        st.header("🎤 Vocal Lab")

        # 보컬 파일 확인
        vocal_path = None
        if 'stems' in st.session_state and st.session_state['stems'].get('vocals'):
            vocal_path = st.session_state['stems'].get('vocals')
            st.success("✅ Using separated **Vocals** stem.")
        else:
            st.warning("⚠️ No separated vocals found. Using original track (Accuracy may be lower).")
            vocal_path = temp_path

        col_lyric, col_timbre = st.columns(2)

        with col_lyric:
            st.subheader("📝 Lyrics (Whisper)")
            if st.button("Transcribe Lyrics"):
                with st.spinner("Transcribing..."):
                    from transcribe import transcribe_audio

                    text, lang = transcribe_audio(vocal_path)
                    if text:
                        st.text_area("Result", text, height=200)

        with col_timbre:
            st.subheader("🎨 Timbre Analysis")
            st.caption(f"Tags: {', '.join(TIMBRE_TAGS)}")

            if st.button("Analyze Timbre"):
                with st.spinner("Listening to Timbre..."):
                    result = analyze_timbre(timbre_model, vocal_path)

                    if "Error" in result:
                        st.error(f"Error: {result['Error']}")
                    else:
                        st.write("### Analysis Result")
                        for tag, prob in result.items():
                            val = int(prob * 100)
                            st.progress(val, text=f"**{tag}**: {val}%")

    # === TAB 5: Discovery ===
    with tab5:
        st.header("🔭 Similar Song Discovery")
        st.info("Finds songs with acoustically similar vibe using Siamese Network embeddings.")

        if not db_vectors:
            st.error("❌ Database is empty. Please run `src/preprocess.py` to index GTZAN data.")

        else:
            if st.button("🔎 Find Similar Tracks"):
                with st.spinner("Calculating Embeddings & Matching..."):
                    results = find_similar_songs(temp_path, siamese_model, db_vectors, top_k=5)

                    if results:
                        st.subheader("Top 5 Matches")
                        for idx, (name, score) in enumerate(results):
                            sim_percent = int(score * 100)

                            with st.container():
                                c1, c2 = st.columns([1, 4])
                                with c1:
                                    st.metric(label="Similarity", value=f"{sim_percent}%")
                                with c2:
                                    st.markdown(f"**{idx + 1}. {name}**")
                                    if "jazz" in name:
                                        st.caption("🎷 Jazz Vibe detected")
                                    elif "hiphop" in name:
                                        st.caption("🎤 Hiphop Vibe detected")
                                    elif "rock" in name:
                                        st.caption("🎸 Rock Vibe detected")

                                st.divider()
                    else:
                        st.warning("No matches found.")