import librosa
import numpy as np
import scipy.stats


def extract_advanced_features(y, sr):
    """
    Advanced feature extraction function based on signal processing,
    not just simple RMS/BPM.
    Returns: dict (0.0 ~ 1.0 normalized values)
    """
    # 1. Energy
    # Combination of RMS (loudness) + Spectral Centroid (timbral brightness)
    rms = librosa.feature.rms(y=y)[0]
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    # Normalization (heuristic normalization)
    norm_rms = np.mean(rms) / 0.25  # Approximate max RMS
    norm_cent = np.mean(spec_cent) / 4000
    energy = (0.6 * norm_rms) + (0.4 * norm_cent)

    # 2. Danceability
    # Measures the strength of the beat (Pulse Clarity) using a Tempogram
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)

    # Higher variance in the tempogram indicates a clearer beat
    beat_strength = np.mean(np.max(tempogram, axis=0))
    danceability = np.clip(beat_strength / 3.0, 0, 1)

    # 3. Acousticness
    # Lower Spectral Flatness (clearer tone) suggests a higher probability of being acoustic
    # (The opposite implies noise/distortion)
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    acousticness = 1.0 - np.mean(flatness)  # High flatness (1.0) suggests machine sound/noise

    # 4. Valence (Mood/Cheerfulness)
    # Difficult, but lower harmonic complexity (Spectral Contrast) tends to correlate with a brighter mood
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    valence = 1.0 - (np.mean(contrast) / 40.0)  # Higher contrast (more complex) yields a lower valence

    # Clip values (force 0 to 1 range)
    features = {
        "energy": float(np.clip(energy, 0, 1)),
        "danceability": float(np.clip(danceability, 0, 1)),
        "acousticness": float(np.clip(acousticness, 0, 1)),
        "valence": float(np.clip(valence, 0, 1))
    }
    return features