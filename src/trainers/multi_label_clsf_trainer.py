import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.metrics import mAP


class MultiLabelClassificationTrainer:
    def __init__(self, model, opt, train_loader, val_loader, device):
        self.model = model.to(device)
        self.opt = opt
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def train_loop(self, num_epochs):
        for i in range(1, num_epochs + 1):
            print(f'Epoch {i}:')
            self.train_epoch()
            self.validation()
            print('-' * 100)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_map = 0
        for x, y in tqdm(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            output = self.model(x)
            loss = F.binary_cross_entropy_with_logits(output, y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            total_loss += loss.detach()
            total_map += mAP(output.detach(), y)
        print('Train loss:', total_loss / len(self.train_loader))
        print('Train mAP:', total_map / len(self.train_loader))

    @torch.no_grad()
    def validation(self):
        self.model.eval()
        total_loss = 0
        total_map = 0
        for x, y in tqdm(self.val_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            output = self.model(x)
            loss = F.binary_cross_entropy_with_logits(output, y)

            total_loss += loss
            total_map += mAP(output, y)
        print('Validation loss:', total_loss / len(self.val_loader))
        print('Validation mAP:', total_map / len(self.val_loader))
