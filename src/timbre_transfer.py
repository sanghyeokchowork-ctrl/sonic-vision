import torch
import torchaudio
import torchaudio.functional as F
import numpy as np


class TimbreSimulator:
    def __init__(self, sample_rate=44100):  # [Change] 22050 -> 44100 (CD quality)
        self.sr = sample_rate

    def apply_style(self, waveform, style):
        """
        High-Quality DSP Processing
        Input: (Channels, Time) Tensor
        """
        # Convert to Tensor
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform)

        # Check dimensions (expect (1, Time))
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # DSP based on style (gain values slightly adjusted for natural sound)
        if style == "Bright":
            # High Shelf: Boost above 6kHz
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
        High-quality conversion for Streamlit playback (securing headroom)
        """
        audio_np = waveform.detach().cpu().numpy()

        if audio_np.ndim > 1:
            audio_np = audio_np.flatten()

        # Peak normalization (prevents clipping/distortion)
        max_val = np.abs(audio_np).max()
        if max_val > 0:
            # Set to 0.95 to leave a little headroom
            audio_np = audio_np / max_val * 0.95

        return audio_np