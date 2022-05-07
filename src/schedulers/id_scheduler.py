from torch.optim.lr_scheduler import LambdaLR


class IdScheduler(LambdaLR):
    def __init__(self, opt, n_iters):
        super().__init__(opt, lambda _: 1)
