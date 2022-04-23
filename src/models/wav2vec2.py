from typing import List

import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from src.datasets import SSPNetVCEvent


class Wav2Vec2Pretrained(torch.nn.Module):
    def __init__(self, size='base'):
        super().__init__()
        if size not in ('base', 'large'):
            raise ValueError('Invalid size')
        model_name = f'facebook/wav2vec2-{size}'
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

    def collate(self, wavs: List[np.ndarray]) -> torch.Tensor:
        return self.processor(
            wavs, padding=True, sampling_rate=16_000,
            pad_to_multiple_of=128, return_tensors='pt'
        ).input_values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).last_hidden_state

    @staticmethod
    def align(wav: torch.Tensor, events: List[SSPNetVCEvent]) -> torch.Tensor:
        if wav.ndim != 1:
            raise ValueError('Expected one channel wav')
        num_emb = 1 + (len(wav) - 400) // 320
        labels = torch.zeros((num_emb,), dtype=torch.int)
        for event in events:
            start_emb_idx = (int(event.start * 16_000) + 120) // 320
            end_emb_idx = (int(event.end * 16_000) + 120) // 320
            labels[start_emb_idx:end_emb_idx] = event.label_idx
        return labels