import numpy as np
import torch

from src.datasets import Event
from src.models import Spectrogrammer


def test_spectrogrammer_aligner():
    wav = torch.zeros((2048,))
    s = Spectrogrammer(16_000, 1024, 512, 64)
    events = [
        Event('asdf', 2, 0, 0.031),
        Event('abobus', 1, 0.048, 0.065),
        Event('sas', 12, 0.095, 0.12)
    ]
    expected = torch.tensor([0, 1, 12], dtype=torch.long)
    assert (s.align(wav, events) == expected).all()


def test_spectrogrammer_collator():
    wavs = [np.zeros(10 * n) for n in range(20)]
    s = Spectrogrammer(16_000, 1024, 512, 64)
    wavs = s.collate(wavs)
    assert (wavs == torch.zeros(20, 190)).all()
