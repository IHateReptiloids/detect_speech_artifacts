import torch


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
