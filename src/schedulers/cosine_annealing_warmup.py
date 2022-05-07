from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR


class CosineAnnealingWarmupScheduler:
    def __init__(self, opt, n_warmup, n_iters):
        self.cur_iter = 0
        self.n_warmup = n_warmup

        self.cosine_scheduler = CosineAnnealingLR(opt, n_iters - n_warmup)
        self.warmup_scheduler = LambdaLR(opt, lambda x: x / n_warmup)

    def step(self):
        self.cur_iter += 1
        if self.cur_iter <= self.n_warmup:
            self.warmup_scheduler.step()
        else:
            self.cosine_scheduler.step()
