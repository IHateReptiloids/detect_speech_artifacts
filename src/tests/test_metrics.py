import torch

from src.metrics import mAP


def test_mAP():
    logits = torch.tensor([[2., 1., 3.],
                           [0.5, 0.3, -1234.],
                           [0.1, 0.2, 0.3]])
    targets = torch.tensor([[1., 0., 0.],
                            [1., 1., 1.],
                            [1., 1., 0.]])
    expected = torch.tensor([25 / 36])
    assert torch.allclose(mAP(logits, targets), expected)
