from typing import List

import numpy as np
import torch
import torch.nn as nn

from .spectrogrammer import Spectrogrammer
from src.datasets import Event
from src.utils import align


class CRNN(nn.Module):
    INPUT_SR = 16_000

    def __init__(self, n_fft, hop_length, n_mels, conv_out_c,
                 conv_kernel_size, conv_stride, gru_num_features,
                 gru_num_layers, gru_dropout, num_classes):
        super().__init__()

        self.spectrogrammer = Spectrogrammer(
            sample_rate=self.INPUT_SR,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

        # conv expects spectrogram of shape [B, 1, T, F] as input
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=conv_out_c,
            kernel_size=conv_kernel_size, stride=conv_stride
        )

        conv_out_freq = (n_mels - conv_kernel_size[1]) // conv_stride[1] + 1

        self.gru = nn.GRU(
            input_size=(conv_out_c * conv_out_freq),
            hidden_size=gru_num_features,
            num_layers=gru_num_layers,
            dropout=gru_dropout,
            batch_first=True
        )

        self.head = nn.Linear(gru_num_features, num_classes)
    
    def align(self, wav: torch.Tensor, events: List[Event]) -> torch.Tensor:
        win_length = self.spectrogrammer.win_length +\
            (self.conv.kernel_size[0] - 1) * self.spectrogrammer.hop_length
        hop_length = self.conv.stride[0] * self.spectrogrammer.hop_length
        return align(wav, events, self.spectrogrammer.sample_rate,
                     win_length, hop_length)
    
    def collate(self, wavs: List[np.ndarray]) -> torch.Tensor:
        return self.spectrogrammer.collate(wavs)

    def forward(self, x):
        x = self.spectrogrammer(x)
        x = self.conv(x.unsqueeze(dim=1)).transpose(1, 2)
        # x has shape [B, T, C, F] now
        x = torch.flatten(x, start_dim=2)
        x = self.gru(x)[0]
        return self.head(x)
