from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb


class UnsupervisedFineTuningTrainer:
    def __init__(self, model, opt, device):
        self.model = model.to(device)
        self.opt = opt
        self.device = device

        self._num_iter = 0
    
    def collate_fn(self, objects):
        wavs, labels = list(zip(*objects))
        wavs = self.model[0].collate(wavs)
        labels = torch.stack([self.model[0].align(wav, label)
                              for wav, label in zip(wavs, labels)], dim=0)
        return wavs, labels
    
    def load_state_dict(self, sd):
        self.model.load_state_dict(sd['model'])
        self.opt.load_state_dict(sd['opt'])
        self._num_iter = sd['num_iter']
    
    def state_dict(self):
        sd = OrderedDict()
        sd['model'] = self.model.state_dict()
        sd['opt'] = self.opt.state_dict()
        sd['num_iter'] = self._num_iter
        return sd

    def train_loop(self, num_epochs, train_loader, val_loader,
                   checkpoint_dir, checkpoint_freq):
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, num_epochs + 1):
            print(f'Epoch {i}:')
            self.train_epoch(train_loader)
            self.validation(val_loader)
            print('-' * 100)
            if checkpoint_dir is not None and i % checkpoint_freq == 0:
                checkpoint_path = checkpoint_dir / f'{i}.pth'
                print(f'Saving checkpoint to {checkpoint_path}')
                torch.save(self.state_dict(), checkpoint_path)
                wandb.save(str(checkpoint_path))

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_acc = 0
        for x, y in tqdm(train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            output = self.model(x)
            loss = F.cross_entropy(output.view(-1, output.shape[-1]),
                                   y.flatten())
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            total_loss += loss.detach()
            acc = (output.argmax(dim=-1) == y).float().mean()
            total_acc += acc

            wandb.log(
                data={'train/loss': loss.detach(), 'train/accuracy': acc},
                step=self._num_iter,
                commit=(self._num_iter % 10 == 0)
            )
            self._num_iter += 1
        print('Train loss:', total_loss / len(train_loader))
        print('Train accuracy:', total_acc / len(train_loader))

    @torch.no_grad()
    def validation(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_acc = 0
        for x, y in tqdm(val_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            output = self.model(x)
            loss = F.cross_entropy(output.view(-1, output.shape[-1]),
                                   y.flatten())

            total_loss += loss
            acc = (output.argmax(dim=-1) == y).float().mean()
            total_acc += acc

        val_loss = total_loss / len(val_loader)
        val_acc = total_acc / len(val_loader)
        wandb.log(
            data={'val/loss': val_loss, 'val/accuracy': val_acc},
            step=self._num_iter
        )
        print('Validation loss:', total_loss / len(val_loader))
        print('Validation accuracy:', total_acc / len(val_loader))
