import numpy as np
import librosa
import os


class MixingEngineer:
    def __init__(self, sample_rate=22050):
        self.sr = sample_rate

        # Frequency Bands definition
        self.BANDS = {
            "Sub Bass": (20, 60),
            "Bass": (60, 250),
            "Low Mids": (250, 500),
            "Mids": (500, 2000),
            "High Mids": (2000, 4000),
            "Presence": (4000, 6000),
            "Brilliance": (6000, 20000)
        }

        # Ideal Balance Profile per Genre (Relative Energy Ratios)
        # Higher values mean that band should be emphasized
        self.GENRE_TARGETS = {
            'hiphop': {'Sub Bass': 1.2, 'Bass': 1.1, 'Brilliance': 1.1},  # Emphasize Kick/Bass & Hi-hats
            'pop': {'Bass': 1.0, 'Mids': 1.1, 'Presence': 1.1},  # Emphasize Vocals (Mids/Presence)
            'rock': {'Low Mids': 1.1, 'Mids': 1.2, 'High Mids': 1.1},  # Guitar & Snare body/punch
            'jazz': {'Bass': 1.0, 'Low Mids': 1.0, 'Brilliance': 0.8},  # Warm and smooth tone
            'classical': {'Sub Bass': 0.8, 'Brilliance': 0.9},  # Dynamic and natural tone
            'default': {'Sub Bass': 1.0, 'Bass': 1.0, 'Mids': 1.0, 'Brilliance': 1.0}
        }

    def analyze_frequency_balance(self, y):
        """
        Analyzes the energy of each frequency band using FFT.
        """
        # Short-Time Fourier Transform
        spec = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=self.sr)

        # Sum of total energy (for normalization)
        total_energy = np.sum(spec)
        if total_energy == 0: return {}

        band_energies = {}

        for band_name, (low_f, high_f) in self.BANDS.items():
            # Find the indices of the Bins corresponding to the frequency band
            idx = np.where((freqs >= low_f) & (freqs <= high_f))[0]
            if len(idx) > 0:
                # Calculate the average energy of the band
                avg_energy = np.mean(spec[idx, :])
                band_energies[band_name] = avg_energy
            else:
                band_energies[band_name] = 0

        # Normalize values (convert to a ratio relative to the overall average)
        mean_val = np.mean(list(band_energies.values()))
        normalized_energies = {k: v / mean_val for k, v in band_energies.items()}

        return normalized_energies

    def get_mixing_suggestions(self, file_path, detected_genre='pop'):
        """
        Analyzes the audio and suggests EQ/Gain adjustments suitable for the genre.
        """
        try:
            y, _ = librosa.load(file_path, sr=self.sr, duration=60)  # Analyze the first 60 seconds
        except Exception as e:
            return {"error": str(e)}

        # 1. Analyze the current track's balance
        current_balance = self.analyze_frequency_balance(y)

        # 2. Get the target for the detected genre (use default if not specified)
        target = self.GENRE_TARGETS.get(detected_genre, self.GENRE_TARGETS['default'])

        suggestions = []
        alert_level = "Green"  # Green, Yellow, Red

        # 3. Compare and generate suggestions
        print(f"\n📊 Mixing Analysis for [{detected_genre.upper()}] style:")

        for band, current_val in current_balance.items():
            # Bands without an explicit target value default to 1.0
            target_val = target.get(band, 1.0)

            # Calculate the ratio difference
            ratio = current_val / target_val

            # Set threshold (to avoid being too sensitive)
            # ratio > 1.2 : Too high (Requires Cut)
            # ratio < 0.8 : Too low (Requires Boost)

            if ratio > 1.25:
                dB = 20 * np.log10(ratio)  # Approximate dB conversion
                suggestions.append(f"🔻 **Cut {band}**: -{dB:.1f}dB (Too Boomy/Harsh)")
                if dB > 3: alert_level = "Red"

            elif ratio < 0.75:
                dB = abs(20 * np.log10(ratio))
                suggestions.append(f"🔺 **Boost {band}**: +{dB:.1f}dB (Lacking energy)")
                if dB > 3: alert_level = "Red"

        # 4. Simple check of Dynamic Range (similar to LUFS/RMS concept)
        rms = librosa.feature.rms(y=y)[0]
        peak = np.max(np.abs(y))
        crest_factor = 20 * np.log10(peak / np.mean(rms))

        dynamic_advice = ""
        if crest_factor < 6:
            dynamic_advice = "⚠️ Track is very compressed (Loudness War?). Reduce Limiter."
        elif crest_factor > 14:
            dynamic_advice = "ℹ️ High Dynamic Range. Consider using a Compressor."
        else:
            dynamic_advice = "✅ Dynamic Range is healthy."

        if not suggestions:
            suggestions.append("✅ Frequency balance looks great for this genre!")

        return {
            "suggestions": suggestions,
            "balance_data": current_balance,
            "dynamic_advice": dynamic_advice,
            "alert_level": alert_level
        }


if __name__ == "__main__":
    # Test Code
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # Test file path (Change this to a file that exists for testing)
    test_song = os.path.join(project_root, "data", "my_songs", "9624 JAZZ CLUB AR.wav")

    if os.path.exists(test_song):
        engineer = MixingEngineer()

        # Test with a hypothetical genre 'hiphop'
        result = engineer.get_mixing_suggestions(test_song, detected_genre='hiphop')

        print("\n💡 AI Mixing Tips:")
        for tip in result['suggestions']:
            print(tip)
        print(f"\n{result['dynamic_advice']}")
    else:
        print("❌ Test file not found.")