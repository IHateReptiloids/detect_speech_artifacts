from typing import List

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from .head import Head
from src.datasets import Event
from src.utils import align


class Wav2Vec2Pretrained(torch.nn.Module):
    INPUT_SR = 16_000

    WIN_LENGTH = 400
    HOP_LENGTH = 320

    def __init__(self, cfg, num_classes=None):
        super().__init__()
        args = cfg.model.args
        if args.size not in ('base', 'large'):
            raise ValueError('Invalid size')
        model_name = f'facebook/wav2vec2-{args.size}'
        self.num_features = 768 if args.size == 'base' else 1024
        self.model = Wav2Vec2Model.from_pretrained(model_name)

        if args.freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

        self.head = None
        if num_classes is not None:
            self.head = Head(cfg.model.head, self.num_features, num_classes)

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

    def collate(self, wavs: List[np.ndarray]) -> torch.Tensor:
        return self.processor(
            wavs, padding=True, sampling_rate=self.INPUT_SR,
            pad_to_multiple_of=128, return_tensors='pt'
        ).input_values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x).last_hidden_state
        if self.head is not None:
            x = self.head(x)
        return x

    @classmethod
    def align(cls, wav: torch.Tensor, events: List[Event]) -> torch.Tensor:
        return align(wav, events, cls.INPUT_SR, cls.WIN_LENGTH, cls.HOP_LENGTH)
