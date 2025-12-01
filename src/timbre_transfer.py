import torch
import torchaudio
import torchaudio.functional as F
import numpy as np


class TimbreSimulator:
    def __init__(self, sample_rate=44100):  # [변경] 22050 -> 44100 (CD 음질)
        self.sr = sample_rate

    def apply_style(self, waveform, style):
        """
        High-Quality DSP Processing
        Input: (Channels, Time) Tensor
        """
        # Tensor 변환
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform)

        # 차원 확인 (1, Time)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # 스타일별 DSP (게인 값을 조금 더 자연스럽게 조정)
        if style == "Bright":
            # High Shelf: 6kHz 이상 부스트
            processed = F.highpass_biquad(waveform, self.sr, cutoff_freq=2000.0, Q=0.7)
            processed = F.equalizer_biquad(processed, self.sr, center_freq=8000, gain=6.0, Q=0.707)
            return processed

        elif style == "Warm":
            # Low-Mid Boost (Tube warmth simulation)
            processed = F.equalizer_biquad(waveform, self.sr, center_freq=400, gain=5.0, Q=0.8)
            processed = F.lowpass_biquad(processed, self.sr, cutoff_freq=5000.0)
            return processed

        elif style == "Breathy":
            # Air Band (12kHz+) Boost
            processed = F.highpass_biquad(waveform, self.sr, cutoff_freq=1000.0)
            processed = F.equalizer_biquad(processed, self.sr, center_freq=12000, gain=9.0, Q=0.5)
            return processed

        elif style == "Rough":
            # Hard Clipping (Distortion)
            gain = 8.0
            processed = torch.clamp(waveform * gain, min=-0.9, max=0.9)
            processed = processed / gain * 1.2
            return processed

        elif style == "Clean":
            # Vocal Scoop (Remove Mud at 300Hz) + Presence
            processed = F.equalizer_biquad(waveform, self.sr, center_freq=350, gain=-6.0, Q=1.2)
            processed = F.equalizer_biquad(processed, self.sr, center_freq=4000, gain=3.0, Q=0.7)
            return processed

        else:
            return waveform

    def tensor_to_numpy(self, waveform):
        """
        Streamlit 재생을 위한 고음질 변환 (Headroom 확보)
        """
        audio_np = waveform.detach().cpu().numpy()

        if audio_np.ndim > 1:
            audio_np = audio_np.flatten()

        # 피크 정규화 (소리 깨짐 방지)
        max_val = np.abs(audio_np).max()
        if max_val > 0:
            # 0.95로 설정하여 헤드룸을 약간 남김
            audio_np = audio_np / max_val * 0.95

        return audio_np