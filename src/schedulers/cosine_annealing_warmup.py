from collections import OrderedDict

from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR


class CosineAnnealingWarmupScheduler:
    def __init__(self, opt, n_warmup, n_iters):
        self.cur_iter = 0
        self.n_warmup = n_warmup

        self.cosine_scheduler = CosineAnnealingLR(opt, n_iters - n_warmup)
        self.warmup_scheduler = LambdaLR(opt, lambda x: x / n_warmup)
    
    def load_state_dict(self, sd):
        self.cur_iter = sd['cur_iter']
        self.n_warmup = sd['n_warmup']
        self.cosine_scheduler.load_state_dict(sd['cosine_scheduler'])
        self.warmup_scheduler.load_state_dict(sd['warmup_scheduler'])

    def state_dict(self):
        sd = OrderedDict()
        sd['cur_iter'] = self.cur_iter
        sd['n_warmup'] = self.n_warmup
        sd['cosine_scheduler'] = self.cosine_scheduler.state_dict()
        sd['warmup_scheduler'] = self.warmup_scheduler.state_dict()
        return sd

    def step(self):
        self.cur_iter += 1
        if self.cur_iter <= self.n_warmup:
            self.warmup_scheduler.step()
        else:
            self.cosine_scheduler.step()
