from typing import List

import numpy as np
import torch
import torch.nn as nn

from .spectrogrammer import Spectrogrammer
from src.datasets import Event


class SubSpectralNorm(nn.BatchNorm2d):
    def __init__(self, num_channels, num_subbands):
        super().__init__(num_channels * num_subbands)
        self.num_subbands = num_subbands
    
    def forward(self, x):
        N, C, F, T = x.shape
        if C * self.num_subbands != self.num_features:
            raise ValueError(f'Expected {self.num_features} channels, got {C}')
        if F % self.num_subbands != 0:
            raise ValueError(
                f'num_subbands must divide number\
                  of frequency bins, got {F=} {self.num_subbands=}'
            )
        x = x.view(N, C * self.num_subbands, F // self.num_subbands, T)
        x = super().forward(x)
        x = x.view(N, C, F, T)
        return x


class BCConvBlock(nn.Module):
    def __init__(self, in_c, out_c, f_ks, t_ks, f_stride, t_stride, f_dilation,
                 t_dilation, num_subbands, dropout):
        super().__init__()

        self.transition = None
        if in_c != out_c:
            self.transition = nn.Conv2d(in_c, out_c, kernel_size=(1, 1))

        self.f1 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=(1, t_ks), stride=(1, t_stride),
                      dilation=(1, t_dilation), groups=out_c,
                      padding=(0, ((t_ks - 1) * t_dilation) // 2)),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=(1, 1)),
            nn.Dropout2d(dropout, inplace=True)
        )

        self.f2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=(f_ks, 1), stride=(f_stride, 1),
                      dilation=(f_dilation, 1), groups=out_c,
                      padding=(((f_ks - 1) * f_dilation) // 2, 0)),
            SubSpectralNorm(out_c, num_subbands)
        )
    
    def forward(self, x):
        # x has shape [N, C, F, T]
        if self.transition is not None:
            x = self.transition(x)
        f2 = self.f2(x)
        f1 = self.f1(f2.mean(dim=2, keepdim=True))
        result = f1 + f2
        if self.transition is None:
            result = result + x
        return result


class BCConvBlocks(nn.Module):
    def __init__(self, n, mult, in_c, out_c, f_ks, t_ks, f_stride, t_stride,
                 f_dilation, t_dilation, num_subbands, dropout):
        super().__init__()

        layers = []
        for _ in range(n):
            layers.append(
                BCConvBlock(
                    int(in_c * mult), int(out_c * mult), f_ks=f_ks, t_ks=t_ks,
                    f_stride=f_stride, t_stride=t_stride,
                    f_dilation=f_dilation, t_dilation=t_dilation,
                    num_subbands=num_subbands, dropout=dropout
                )
            )
            in_c = out_c
            f_stride = 1
            t_stride = 1

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class BCResNet(nn.Module):
    INPUT_SR = 16_000

    def __init__(self, n_fft, hop_length, n_mels, mult, num_classes):
        super().__init__()

        if n_mels != 40:
            raise ValueError('Only n_mels=40 is supported')

        self.spectrogrammer = Spectrogrammer(
            sample_rate=self.INPUT_SR,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

        self.mult = mult

        self.net = nn.Sequential(
            nn.Conv2d(1, 16 * mult, kernel_size=(5, 5), stride=(2, 1),
                      padding=(2, 2)),
            BCConvBlocks(2, mult, 16, 8, f_ks=3, t_ks=3, f_stride=1,
                         t_stride=1, f_dilation=1, t_dilation=1,
                         num_subbands=5, dropout=0.1),
            BCConvBlocks(2, mult, 8, 12, f_ks=3, t_ks=3, f_stride=2,
                         t_stride=1, f_dilation=1, t_dilation=2,
                         num_subbands=5, dropout=0.1),
            BCConvBlocks(4, mult, 12, 16, f_ks=3, t_ks=3, f_stride=2,
                         t_stride=1, f_dilation=1, t_dilation=4,
                         num_subbands=5, dropout=0.1),
            BCConvBlocks(4, mult, 16, 20, f_ks=3, t_ks=3, f_stride=1,
                         t_stride=1, f_dilation=1, t_dilation=8,
                         num_subbands=5, dropout=0.1),
            nn.Conv2d(20 * mult, 20 * mult, kernel_size=(5, 5),
                      groups=(20 * mult), padding=(0, 2)),
            nn.Conv2d(20 * mult, num_classes, kernel_size=(1, 1))
        )

    def align(self, wav: torch.Tensor, events: List[Event]) -> torch.Tensor:
        return self.spectrogrammer.align(wav, events)
    
    def collate(self, wavs: List[np.ndarray]) -> torch.Tensor:
        return self.spectrogrammer.collate(wavs)

    def forward(self, x):
        spec = self.spectrogrammer(x).transpose(-1, -2)
        return self.net(spec.unsqueeze(dim=1)).squeeze(dim=2).transpose(-1, -2)
