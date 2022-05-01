import numpy as np
import torch

from src.datasets import Event, SSPNetVC
from src.models import Wav2Vec2Pretrained


def test_wav2vec2_aligner():
    wav = torch.zeros((1359,))
    events = [Event('filler', SSPNetVC.LABEL2IND['filler'], 0, 0.015),
              Event('laughter', SSPNetVC.LABEL2IND['laughter'], 0.019, 0.033),
              Event('filler', SSPNetVC.LABEL2IND['filler'], 0.034, 0.052)]
    expected = torch.tensor([1, 2, 0], dtype=torch.int)
    assert (Wav2Vec2Pretrained.align(wav, events) == expected).all()


def test_wav2vec2_collator():
    wavs = [np.zeros(10 * n) for n in range(20)]
    w2v = Wav2Vec2Pretrained()
    wavs = w2v.collate(wavs)
    assert (wavs == torch.zeros(20, 256)).all()
