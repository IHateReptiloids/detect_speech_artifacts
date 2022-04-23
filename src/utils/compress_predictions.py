import torch


def compress_predictions(predictions: torch.Tensor):
    if predictions.dim() != 1:
        raise ValueError('predictions tensor must be one-dimensional')
    l, r = 0, 0
    res = []
    while l < len(predictions):
        while r < len(predictions) and predictions[l] == predictions[r]:
            r += 1
        res.append((predictions[l].item(), l, r))
        l = r
    return res
