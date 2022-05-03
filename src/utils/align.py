from typing import List

import torch

from src.datasets import Event


def align(wav: torch.Tensor, events: List[Event],
          sample_rate: int, win_length: int, hop_length: int) -> torch.Tensor:
    if wav.ndim != 1:
        raise ValueError('Expected one channel wav')
    num_emb = 1 + (len(wav) - win_length) // hop_length
    if num_emb <= 0:
        raise ValueError('wav is too short')
    labels = torch.zeros((num_emb,), dtype=torch.long)
    for event in events:
        start_emb_idx = time2idx(event.start, sample_rate,
                                 win_length, hop_length)
        end_emb_idx = time2idx(event.end, sample_rate,
                               win_length, hop_length)
        labels[start_emb_idx:end_emb_idx] = event.label_idx
    return labels


def time2idx(t: float, sample_rate: int,
             win_length: int, hop_length: int) -> int:
    """
    returns index of leftmost frame whose window
    center is further than specified t
    """
    return max(0, 1 + (int(t * sample_rate) - win_length // 2) // hop_length)
