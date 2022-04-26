import torch

from src.metrics import f1_score, mAP


def test_f1_score():
    pred = torch.tensor([0, 0, 1, 2, 2])
    true = torch.tensor([0, 1, 1, 2, 0])
    expected = torch.tensor([0.5, 2 / 3, 2 / 3])
    assert torch.allclose(f1_score(pred, true, 3), expected)


def test_mAP():
    logits = torch.tensor([[2., 1., 3.],
                           [0.5, 0.3, -1234.],
                           [0.1, 0.2, 0.3]])
    targets = torch.tensor([[1., 0., 0.],
                            [1., 1., 1.],
                            [1., 1., 0.]])
    expected = torch.tensor([25 / 36])
    assert torch.allclose(mAP(logits, targets), expected)
