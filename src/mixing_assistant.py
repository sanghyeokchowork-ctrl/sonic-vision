import numpy as np
import librosa
import os


class MixingEngineer:
    def __init__(self, sample_rate=22050):
        self.sr = sample_rate

        # 주파수 대역 정의 (Frequency Bands)
        self.BANDS = {
            "Sub Bass": (20, 60),
            "Bass": (60, 250),
            "Low Mids": (250, 500),
            "Mids": (500, 2000),
            "High Mids": (2000, 4000),
            "Presence": (4000, 6000),
            "Brilliance": (6000, 20000)
        }

        # 장르별 이상적인 밸런스 프로필 (Relative Energy Ratios)
        # 값이 높을수록 해당 대역이 강조되어야 함
        self.GENRE_TARGETS = {
            'hiphop': {'Sub Bass': 1.2, 'Bass': 1.1, 'Brilliance': 1.1},  # 킥/베이스 & 하이햇 강조
            'pop': {'Bass': 1.0, 'Mids': 1.1, 'Presence': 1.1},  # 보컬(Mids/Presence) 강조
            'rock': {'Low Mids': 1.1, 'Mids': 1.2, 'High Mids': 1.1},  # 기타 & 스네어 바디감
            'jazz': {'Bass': 1.0, 'Low Mids': 1.0, 'Brilliance': 0.8},  # 따뜻하고 부드러운 톤
            'classical': {'Sub Bass': 0.8, 'Brilliance': 0.9},  # 다이내믹하고 자연스러운 톤
            'default': {'Sub Bass': 1.0, 'Bass': 1.0, 'Mids': 1.0, 'Brilliance': 1.0}
        }

    def analyze_frequency_balance(self, y):
        """
        FFT를 사용하여 주파수 대역별 에너지를 분석합니다.
        """
        # Short-Time Fourier Transform
        spec = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=self.sr)

        # 전체 에너지 합계 (Normalize를 위해)
        total_energy = np.sum(spec)
        if total_energy == 0: return {}

        band_energies = {}

        for band_name, (low_f, high_f) in self.BANDS.items():
            # 해당 주파수 대역에 해당하는 Bin의 인덱스 찾기
            idx = np.where((freqs >= low_f) & (freqs <= high_f))[0]
            if len(idx) > 0:
                # 해당 대역의 에너지 평균 계산
                avg_energy = np.mean(spec[idx, :])
                band_energies[band_name] = avg_energy
            else:
                band_energies[band_name] = 0

        # 값 정규화 (전체 평균 대비 비율로 변환)
        mean_val = np.mean(list(band_energies.values()))
        normalized_energies = {k: v / mean_val for k, v in band_energies.items()}

        return normalized_energies

    def get_mixing_suggestions(self, file_path, detected_genre='pop'):
        """
        오디오를 분석하고 장르에 맞는 EQ/Gain 조정을 제안합니다.
        """
        try:
            y, _ = librosa.load(file_path, sr=self.sr, duration=60)  # 앞 60초 분석
        except Exception as e:
            return {"error": str(e)}

        # 1. 현재 곡의 밸런스 분석
        current_balance = self.analyze_frequency_balance(y)

        # 2. 목표 장르의 타겟 가져오기 (없으면 default)
        target = self.GENRE_TARGETS.get(detected_genre, self.GENRE_TARGETS['default'])

        suggestions = []
        alert_level = "Green"  # Green, Yellow, Red

        # 3. 비교 및 제안 생성
        print(f"\n📊 Mixing Analysis for [{detected_genre.upper()}] style:")

        for band, current_val in current_balance.items():
            # 타겟값이 명시되지 않은 대역은 기본값 1.0으로 처리
            target_val = target.get(band, 1.0)

            # 비율 차이 계산
            ratio = current_val / target_val

            # Threshold 설정 (너무 민감하지 않게)
            # ratio > 1.2 : 너무 큼 (Cut 필요)
            # ratio < 0.8 : 너무 작음 (Boost 필요)

            if ratio > 1.25:
                dB = 20 * np.log10(ratio)  # 대략적인 dB 환산
                suggestions.append(f"🔻 **Cut {band}**: -{dB:.1f}dB (Too Boomy/Harsh)")
                if dB > 3: alert_level = "Red"

            elif ratio < 0.75:
                dB = abs(20 * np.log10(ratio))
                suggestions.append(f"🔺 **Boost {band}**: +{dB:.1f}dB (Lacking energy)")
                if dB > 3: alert_level = "Red"

        # 4. 다이내믹 레인지 (LUFS/RMS 유사 개념) 간단 체크
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

    # 테스트용 파일 경로 (존재하는 파일로 변경해서 테스트하세요)
    test_song = os.path.join(project_root, "data", "my_songs", "9624 JAZZ CLUB AR.wav")

    if os.path.exists(test_song):
        engineer = MixingEngineer()

        # 가상의 장르 'hiphop'으로 테스트
        result = engineer.get_mixing_suggestions(test_song, detected_genre='hiphop')

        print("\n💡 AI Mixing Tips:")
        for tip in result['suggestions']:
            print(tip)
        print(f"\n{result['dynamic_advice']}")
    else:
        print("❌ Test file not found.")