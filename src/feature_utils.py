import librosa
import numpy as np
import scipy.stats


def extract_advanced_features(y, sr):
    """
    단순 RMS/BPM이 아닌, 신호 처리 기반의 정교한 특징 추출 함수
    Returns: dict (0.0 ~ 1.0 normalized values)
    """
    # 1. Energy (에너지)
    # RMS(음량) + Spectral Centroid(음색의 밝기)를 조합
    rms = librosa.feature.rms(y=y)[0]
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    # 정규화 (heuristic normalization)
    norm_rms = np.mean(rms) / 0.25  # 대략적인 max rms
    norm_cent = np.mean(spec_cent) / 4000
    energy = (0.6 * norm_rms) + (0.4 * norm_cent)

    # 2. Danceability (댄스 적합성)
    # Tempogram을 이용해 비트의 강도(Pulse Clarity)를 측정
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)

    # 비트가 뚜렷할수록 tempogram의 분산이 큼
    beat_strength = np.mean(np.max(tempogram, axis=0))
    danceability = np.clip(beat_strength / 3.0, 0, 1)

    # 3. Acousticness (어쿠스틱함)
    # Spectral Flatness가 낮을수록(톤이 분명할수록) 어쿠스틱할 확률이 높음 (반대는 노이즈/디스토션)
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    acousticness = 1.0 - np.mean(flatness)  # Flatness가 높으면(1.0) 기계음/노이즈

    # 4. Valence (분위기/밝음)
    # 어렵지만, 화성적 복잡도(Spectral Contrast)가 낮을수록 밝은 경향이 있음
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    valence = 1.0 - (np.mean(contrast) / 40.0)  # 대비가 크면(복잡하면) 낮게 책정

    # 값 클리핑 (0~1 사이로 강제)
    features = {
        "energy": float(np.clip(energy, 0, 1)),
        "danceability": float(np.clip(danceability, 0, 1)),
        "acousticness": float(np.clip(acousticness, 0, 1)),
        "valence": float(np.clip(valence, 0, 1))
    }
    return features