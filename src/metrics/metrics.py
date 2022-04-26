import sklearn.metrics
import torch


def f1_score(pred, y, num_classes):
    """
    Computes f1 score for each class
    returns tensor of shape (num_classes, )
    """
    res = torch.empty(num_classes)
    pred = pred.flatten()
    y = y.flatten()
    for i in range(num_classes):
        pred_i = torch.where(pred == i, 1, 0)
        y_i = torch.where(y == i, 1, 0)
        res[i] = sklearn.metrics.f1_score(y_i.cpu().numpy(),
                                          pred_i.cpu().numpy())
    return res.to(pred.device)


def mAP(logits, targets):
    """
    logits: shape (B, C)
    targets: shape (B, C), values from {0, 1}
    returns: mAP
    """
    targets = targets.gather(dim=1, index=torch.argsort(logits, dim=-1,
                                                        descending=True))
    ap = (targets * torch.cumsum(targets, dim=-1) /\
        torch.arange(1, targets.shape[-1] + 1, device=targets.device))\
        .sum(dim=-1) / targets.sum(dim=-1)
    return ap.mean()
