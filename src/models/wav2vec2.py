from multiprocessing.sharedctypes import Value
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from .lstm_wrapper import LSTMWrapper


class Wav2Vec2Pretrained(torch.nn.Module):
    INPUT_SR = 16_000

    def __init__(self, size='base', freeze=True, head=None, num_classes=None):
        super().__init__()
        if size not in ('base', 'large'):
            raise ValueError('Invalid size')
        model_name = f'facebook/wav2vec2-{size}'
        self.num_features = 768 if size == 'base' else 1024
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.head = None

        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
        if head is not None:
            if num_classes is None:
                raise ValueError('You must specify number of classes')
            if head == 'mlp':
                self.head = nn.Linear(self.num_features, num_classes)
            elif head == 'lstm':
                self.head = nn.Sequential(
                    LSTMWrapper(self.num_features, self.num_features,
                                batch_first=True),
                    nn.Linear(self.num_features, num_classes)
                )
            else:
                raise ValueError('Unknown head type')
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

    def collate(self, wavs: List[np.ndarray]) -> torch.Tensor:
        return self.processor(
            wavs, padding=True, sampling_rate=16_000,
            pad_to_multiple_of=128, return_tensors='pt'
        ).input_values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x).last_hidden_state
        if self.head is not None:
            x = self.head(x)
        return x

    @staticmethod
    def align(wav: torch.Tensor, events) -> torch.Tensor:
        if wav.ndim != 1:
            raise ValueError('Expected one channel wav')
        num_emb = 1 + (len(wav) - 400) // 320
        labels = torch.zeros((num_emb,), dtype=torch.long)
        for event in events:
            start_emb_idx = (int(event.start * 16_000) + 120) // 320
            end_emb_idx = (int(event.end * 16_000) + 120) // 320
            labels[start_emb_idx:end_emb_idx] = event.label_idx
        return labels
    
    @staticmethod
    def idx2sec(start: int, end: int) -> Tuple[float, float]:
        return (0.02 * start, 0.02 * (end - 1) + 0.025)
