import streamlit as st
import os
import torch
import numpy as np
import base64
from PIL import Image
import librosa
from torchvision import transforms
import matplotlib.pyplot as plt

# 사용자 모듈 임포트
from model import get_model
from mixing_assistant import MixingEngineer
from vocal_timbre_model import VocalTimbreCNN
from recommend import load_siamese_model, build_database_index, find_similar_songs, audio_to_tensor
from feature_model import FeatureRegressor
from timbre_transfer import TimbreSimulator

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ==========================================
# Configuration & Styling
# ==========================================
st.set_page_config(page_title="Sonic Vision Pro", page_icon="🎵", layout="wide")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

CLASSES = ['blues', 'classical', 'country', 'disco', 'hiphop',
           'jazz', 'metal', 'pop', 'reggae', 'rock']

TIMBRE_TAGS = ['Bright', 'Warm', 'Breathy', 'Rough', 'Clean']

SPOTIFY_GREEN = "#1DB954"
SPOTIFY_RED = "#FF4B4B"
HQ_SR = 44100

# [CSS] 컴팩트 대시보드 + 타이포그래피 & 액션 카드 추가
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Helvetica+Neue:wght@400;700&family=Oswald:wght@700&display=swap');

    /* 1. 기본 설정 */
    .stApp {{
        background-color: #121212 !important;
        color: #FFFFFF !important;
        font-family: 'Helvetica Neue', sans-serif;
    }}
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}

    /* 2. 컴팩트 카드 (Metric Box) - 기존 유지 (90px) */
    .metric-box {{
        background-color: #1E1E1E;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 10px 15px;
        text-align: center;
        height: 90px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        margin-bottom: 5px;
    }}
    .metric-title {{
        color: #888;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 2px;
    }}
    .metric-value {{
        color: {SPOTIFY_GREEN};
        font-size: 24px;
        font-weight: 700;
        line-height: 1.2;
    }}
    .metric-bar {{
        width: 100%;
        height: 4px;
        background-color: #333;
        border-radius: 2px;
        margin-top: 5px;
        overflow: hidden;
    }}
    .metric-fill {{
        height: 100%;
        background-color: {SPOTIFY_GREEN};
    }}

    /* [NEW] 3. Analysis - Genre Hero Design */
    .genre-container {{
        margin-bottom: 5px;
    }}
    .genre-label {{
        color: #888;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0px;
    }}
    .genre-value {{
        font-family: 'Oswald', sans-serif;
        font-size: 64px; /* 타이포그래피 강조 */
        font-weight: 700;
        color: {SPOTIFY_GREEN};
        line-height: 1.1;
        text-transform: uppercase;
        text-shadow: 0 4px 10px rgba(29, 185, 84, 0.2);
    }}
    .genre-sub {{
        color: #CCC;
        font-size: 13px;
        margin-top: 0px;
    }}

    /* [NEW] 4. Mixing - Suggestion Cards */
    .mix-card {{
        background-color: #1E1E1E;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        border-left: 4px solid #555;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        display: flex;
        align-items: center;
    }}
    .mix-card.boost {{ border-left-color: {SPOTIFY_GREEN}; }}
    .mix-card.cut {{ border-left-color: {SPOTIFY_RED}; }}

    .mix-icon {{ font-size: 18px; margin-right: 12px; width: 20px; text-align: center; }}
    .mix-text {{ color: #FFF; font-size: 14px; font-weight: 500; }}
    .mix-sub {{ color: #888; font-size: 11px; margin-left: auto; }}

    /* 5. 텍스트 헤더 여백 제거 */
    h3, h4 {{
        margin-bottom: 0.5rem !important;
        padding-top: 0 !important;
        color: white !important;
    }}

    /* 6. Streamlit 기본 요소 */
    [data-testid="stHeader"] {{ background: transparent; }}

    /* 탭 스타일 */
    button[data-baseweb="tab"] {{
        background-color: transparent !important;
        color: #888 !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {SPOTIFY_GREEN} !important;
        border-color: {SPOTIFY_GREEN} !important;
    }}
</style>
""", unsafe_allow_html=True)


# HTML Card Generator (상단 4개 특징용)
def create_metric_card(label, value_percent):
    return f"""
    <div class="metric-box">
        <div class="metric-title">{label}</div>
        <div class="metric-value">{value_percent}%</div>
        <div class="metric-bar">
            <div class="metric-fill" style="width: {value_percent}%;"></div>
        </div>
    </div>
    """


# [NEW] Genre Hero HTML Generator
def create_genre_hero(genre, confidence):
    return f"""
    <div class="genre-container">
        <div class="genre-label">Detected Genre</div>
        <div class="genre-value">{genre}</div>
        <div class="genre-sub">Confidence: {confidence}%</div>
    </div>
    """


# [NEW] Mixing Card HTML Generator
def create_mixing_card(type, text, subtext=""):
    # type: "boost" or "cut" or "ok"
    icon = "🔼" if type == "boost" else ("🔻" if type == "cut" else "✅")
    css_class = "boost" if type == "boost" else ("cut" if type == "cut" else "")

    return f"""
    <div class="mix-card {css_class}">
        <div class="mix-icon">{icon}</div>
        <div class="mix-text">{text}</div>
        <div class="mix-sub">{subtext}</div>
    </div>
    """


# [Chart] 컴팩트 차트
def plot_compact_chart(data_dict):
    genres = list(data_dict.keys())
    probs = list(data_dict.values())

    fig, ax = plt.subplots(figsize=(5, 2.2))

    fig.patch.set_facecolor('#121212')
    ax.set_facecolor('#121212')

    bars = ax.bar(genres, probs, color=SPOTIFY_GREEN, width=0.6)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis='x', colors='white', labelsize=8, rotation=0)
    ax.tick_params(axis='y', colors='gray', labelsize=7)
    ax.grid(axis='y', color='#333333', linestyle='--', linewidth=0.5)

    return fig


# ==========================================
# Core Logic Loading
# ==========================================
@st.cache_resource
def load_core_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    genre_path = os.path.join(project_root, 'models', 'best_model.pth')
    timbre_path = os.path.join(project_root, 'models', 'vocal_timbre.pth')
    siamese_path = os.path.join(project_root, 'models', 'siamese_net.pth')
    feat_path = os.path.join(project_root, 'models', 'feature_model.pth')
    processed_dir = os.path.join(project_root, 'data', 'processed')

    cls_model = get_model(num_classes=10, device=DEVICE)
    if os.path.exists(genre_path): cls_model.load_state_dict(torch.load(genre_path, map_location=DEVICE))
    cls_model.eval()

    mix_engineer = MixingEngineer(sample_rate=HQ_SR)

    timbre_model = VocalTimbreCNN(num_tags=len(TIMBRE_TAGS)).to(DEVICE)
    if os.path.exists(timbre_path): timbre_model.load_state_dict(torch.load(timbre_path, map_location=DEVICE))
    timbre_model.eval()

    siamese_model = load_siamese_model(siamese_path)
    db_vectors = build_database_index(siamese_model, processed_dir) if os.path.exists(processed_dir) else {}

    feat_model = FeatureRegressor(num_features=4).to(DEVICE)
    if os.path.exists(feat_path): feat_model.load_state_dict(torch.load(feat_path, map_location=DEVICE))
    feat_model.eval()

    timbre_sim = TimbreSimulator(sample_rate=HQ_SR)

    return cls_model, mix_engineer, timbre_model, siamese_model, db_vectors, feat_model, timbre_sim, genre_path, project_root


def analyze_timbre(model, audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=HQ_SR, duration=3.0)
        target_len = HQ_SR * 3
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        input_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

        return dict(zip(TIMBRE_TAGS, probs))
    except Exception as e:
        return {"Error": str(e)}


def predict_track_features(model, audio_path):
    input_tensor = audio_to_tensor(audio_path)
    if input_tensor is None: return None
    with torch.no_grad():
        preds = model(input_tensor).squeeze().cpu().numpy()
    return {"Energy": preds[0], "Danceability": preds[1], "Acousticness": preds[2], "Valence": preds[3]}


# ==========================================
# Main Execution
# ==========================================
st.title("SONIC VISION PRO")
st.caption("The Ultimate AI Engineer for PARFUMDEWALKER")

st.sidebar.header("SYSTEM STATUS")
st.sidebar.success(f"Device: {DEVICE.upper()}")

with st.spinner("Initializing System..."):
    cls_model, mix_engineer, timbre_model, siamese_model, db_vectors, feat_model, timbre_sim, model_path, project_root = load_core_models()

uploaded_file = st.file_uploader("Drop Audio File (MP3/WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if 'genre_probs' not in st.session_state: st.session_state['genre_probs'] = None
    if 'top_genre' not in st.session_state: st.session_state['top_genre'] = 'pop'

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "ANALYSIS", "REMIX", "MIXING", "VOCAL LAB", "DISCOVERY"
    ])

    # === TAB 1: Analysis ===
    with tab1:
        st.subheader("Track DNA")
        from predict import process_audio
        from explain import save_gradcam

        if st.button("START ANALYSIS", use_container_width=True, type="primary"):
            inputs = process_audio(temp_path)
            if inputs is not None:
                with torch.no_grad():
                    outputs = cls_model(inputs)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    avg_probs = torch.mean(probs, dim=0).cpu().numpy()

                st.session_state['genre_probs'] = avg_probs
                top3 = avg_probs.argsort()[-3:][::-1]
                st.session_state['top_genre'] = CLASSES[top3[0]].upper()

                features = predict_track_features(feat_model, temp_path)

                # 1. Deep Audio Features (Compact Row)
                if features:
                    c1, c2, c3, c4 = st.columns(4)
                    with c1: st.markdown(create_metric_card("Energy", int(features['Energy'] * 100)),
                                         unsafe_allow_html=True)
                    with c2: st.markdown(create_metric_card("Danceability", int(features['Danceability'] * 100)),
                                         unsafe_allow_html=True)
                    with c3: st.markdown(create_metric_card("Acousticness", int(features['Acousticness'] * 100)),
                                         unsafe_allow_html=True)
                    with c4: st.markdown(create_metric_card("Valence", int(features['Valence'] * 100)),
                                         unsafe_allow_html=True)

                st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

                # 2. Genre & Heatmap
                col_left, col_right = st.columns([1, 1])

                with col_left:
                    # [변경] create_genre_hero 적용 (큰 글씨)
                    conf = int(avg_probs[top3[0]] * 100)
                    st.markdown(create_genre_hero(st.session_state['top_genre'], conf), unsafe_allow_html=True)

                    chart_data = {CLASSES[i].upper(): avg_probs[i] for i in top3}
                    fig = plot_compact_chart(chart_data)
                    st.pyplot(fig, use_container_width=True)

                with col_right:
                    st.markdown("#### AI Vision")
                    heatmap_path = "temp_heatmap.jpg"
                    try:
                        save_gradcam(temp_path, model_path, heatmap_path)
                        st.image(heatmap_path, use_container_width=True)
                    except:
                        st.warning("Heatmap failed.")
            else:
                st.error("Audio processing failed.")

    # === TAB 2: Remix ===
    with tab2:
        st.header("🎛️ Stem Separation")
        if st.button("SEPARATE STEMS", type="primary"):
            with st.spinner("Processing (Demucs)..."):
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

    # === TAB 3: Mixing ===
    with tab3:
        st.header("🎚️ AI Mixing Assistant")
        target_genre = st.selectbox("Target Style", [c.upper() for c in CLASSES],
                                    index=CLASSES.index(st.session_state['top_genre'].lower()) if st.session_state[
                                                                                                      'top_genre'].lower() in CLASSES else 7)

        if st.button("CHECK BALANCE", type="primary"):
            with st.spinner("Analyzing Spectrum..."):
                mix_result = mix_engineer.get_mixing_suggestions(temp_path, detected_genre=target_genre.lower())
                if "error" in mix_result:
                    st.error(mix_result['error'])
                else:
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        # [변경] 믹싱 카드 디자인 적용
                        st.markdown("#### Action Plan")
                        for tip in mix_result['suggestions']:
                            if "Boost" in tip:
                                parts = tip.split(":")
                                main_text = parts[0].replace("🔺 ", "")
                                sub_text = parts[1] if len(parts) > 1 else ""
                                st.markdown(create_mixing_card("boost", main_text, sub_text), unsafe_allow_html=True)
                            elif "Cut" in tip:
                                parts = tip.split(":")
                                main_text = parts[0].replace("🔻 ", "")
                                sub_text = parts[1] if len(parts) > 1 else ""
                                st.markdown(create_mixing_card("cut", main_text, sub_text), unsafe_allow_html=True)
                            else:
                                st.markdown(create_mixing_card("ok", tip.replace("✅ ", "")), unsafe_allow_html=True)

                    with c2:
                        st.markdown("#### Frequency Spectrum")
                        fig, ax = plt.subplots(figsize=(5, 3))
                        fig.patch.set_facecolor('#121212')
                        ax.set_facecolor('#121212')
                        ax.bar(mix_result['balance_data'].keys(), mix_result['balance_data'].values(),
                               color=SPOTIFY_GREEN)
                        ax.tick_params(axis='x', colors='white', rotation=45, labelsize=8)
                        ax.tick_params(axis='y', colors='gray')
                        for spine in ax.spines.values(): spine.set_visible(False)
                        ax.grid(axis='y', color='#333333', linestyle='--', linewidth=0.5)
                        st.pyplot(fig)

    # === TAB 4: Vocal Lab ===
    with tab4:
        st.header("🎤 Vocal Lab")

        is_separated = False
        vocal_path = None

        if 'stems' in st.session_state and st.session_state['stems'].get('vocals'):
            stem_path = st.session_state['stems'].get('vocals')
            if os.path.exists(stem_path):
                vocal_path = stem_path
                is_separated = True
                st.success("✅ Ready: Separated Vocals Loaded")
            else:
                st.warning("⚠️ Stem files missing. Falling back to original.")
                vocal_path = temp_path
        else:
            st.info("💡 To use Simulation, please **Separate Stems** in the 'Remix' tab first.")
            vocal_path = temp_path

        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown("#### 1. Analysis")
            if st.button("ANALYZE TIMBRE", use_container_width=True):
                with st.spinner("Analyzing..."):
                    result = analyze_timbre(timbre_model, vocal_path)
                    if result:
                        if "Error" in result:
                            st.error(result['Error'])
                        else:
                            for tag, prob in result.items():
                                val = int(prob * 100)
                                st.progress(val, text=f"**{tag}**: {val}%")

        with c2:
            st.markdown("#### 2. Simulation (Preview)")

            if not is_separated:
                st.warning("🚫 Simulation requires separated vocals.")
                st.markdown("1. Go to **Remix** tab.\n2. Click **Separate Stems**.\n3. Come back here!")
            else:
                if 'active_style' not in st.session_state: st.session_state['active_style'] = None


                def style_btn(label, key):
                    type_ = "primary" if st.session_state['active_style'] == key else "secondary"
                    if st.button(label, key=key, use_container_width=True, type=type_):
                        st.session_state['active_style'] = key
                        st.rerun()


                r1_c1, r1_c2, r1_c3 = st.columns(3)
                with r1_c1:
                    style_btn("✨ Bright", "Bright")
                with r1_c2:
                    style_btn("🔥 Warm", "Warm")
                with r1_c3:
                    style_btn("💨 Breathy", "Breathy")

                r2_c1, r2_c2, r2_c3 = st.columns(3)
                with r2_c1:
                    style_btn("🎸 Rough", "Rough")
                with r2_c2:
                    style_btn("🧼 Clean", "Clean")
                with r2_c3:
                    if st.button("↩️ Reset", key="reset", use_container_width=True):
                        st.session_state['active_style'] = None
                        st.rerun()

                try:
                    if st.session_state['active_style']:
                        y, sr = librosa.load(vocal_path, sr=HQ_SR, mono=True)
                        y_tensor = torch.tensor(y).unsqueeze(0)
                        style = st.session_state['active_style']
                        with st.spinner(f"Applying {style}..."):
                            processed = timbre_sim.apply_style(y_tensor, style)
                            audio_data = timbre_sim.tensor_to_numpy(processed)
                            st.audio(audio_data, sample_rate=HQ_SR)
                    else:
                        st.audio(vocal_path, format="audio/wav")
                except Exception as e:
                    st.error(f"Playback Error: {e}")

    # === TAB 5: Discovery ===
    with tab5:
        st.header("🔭 Similar Songs")
        if not db_vectors:
            st.error("Index DB not found.")
        else:
            if st.button("FIND MATCHES", type="primary"):
                with st.spinner("Matching..."):
                    results = find_similar_songs(temp_path, siamese_model, db_vectors)
                    if results:
                        for idx, (name, score) in enumerate(results):
                            st.success(f"**{idx + 1}. {name}** ({int(score * 100)}%)")
                    else:
                        st.warning("No matches.")