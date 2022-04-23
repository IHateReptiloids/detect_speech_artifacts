import torch

from src.utils import compress_predictions


def test_compress_predictions():
    pred = torch.tensor([0, 0, 0, 1, 1, 2, 0, 3, 3])
    expected = [(0, 0, 3), (1, 3, 5), (2, 5, 6), (0, 6, 7), (3, 7, 9)]
    assert compress_predictions(pred) == expected
