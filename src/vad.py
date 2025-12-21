from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class VADConfig:
    backend: str = "silero"   # "silero" or "energy"
    silero_threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 500


class SileroVAD:
    def __init__(self, threshold: float = 0.5, device: str = "cpu"):
        self.threshold = float(threshold)
        self.device = device
        # Offline via torch.hub. First run downloads once.
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        (self.get_speech_timestamps, _, _, _, _) = utils
        self.model.to(self.device).eval()

    def has_speech(self, audio_16k: np.ndarray) -> bool:
        wav = torch.from_numpy(audio_16k.astype(np.float32)).to(self.device)
        ts = self.get_speech_timestamps(
            wav, self.model, sampling_rate=16000, threshold=self.threshold
        )
        return len(ts) > 0


class EnergyVAD:
    def __init__(self, rms_threshold: float = 0.01):
        self.rms_threshold = float(rms_threshold)

    def has_speech(self, audio_16k: np.ndarray) -> bool:
        if audio_16k.size == 0:
            return False
        rms = float(np.sqrt(np.mean(np.square(audio_16k.astype(np.float32))) + 1e-12))
        return rms >= self.rms_threshold


def build_vad(cfg: VADConfig, device: str = "cpu"):
    backend = (cfg.backend or "energy").lower()
    if backend == "silero":
        try:
            return SileroVAD(threshold=cfg.silero_threshold, device=device)
        except Exception:
            # Hard fallback if hub download fails.
            return EnergyVAD(rms_threshold=0.01)
    return EnergyVAD(rms_threshold=0.01)
