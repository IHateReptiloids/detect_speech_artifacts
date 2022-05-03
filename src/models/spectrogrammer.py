from typing import List

import numpy as np
import torch
import torchaudio

from src.datasets import Event
from src.utils import align


class Spectrogrammer(torchaudio.transforms.MelSpectrogram):
    def __init__(self, sample_rate, n_fft, hop_length, n_mels):
        super().__init__(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=False
        )
    
    def forward(self, x):
        return torch.log(super().forward(x).clamp(1e-9, 1e9)).transpose(-2, -1)
    
    def align(self, wav: torch.Tensor, events: List[Event]) -> torch.Tensor:
        return align(wav, events, self.sample_rate,
                     self.win_length, self.hop_length)
    
    def collate(self, wavs: List[np.ndarray]) -> torch.Tensor:
        return torch.nn.utils.rnn.pad_sequence(
            map(torch.from_numpy, wavs),
            batch_first=True
        )
