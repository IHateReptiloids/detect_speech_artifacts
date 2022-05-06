from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from src.metrics import f1_score
from src.utils import visualize


class FramewiseClassificationTrainer:
    def __init__(self, cfg, model_cls, opt_cls, device, train_ds, val_ds):
        self.cfg = cfg
        self.model = model_cls(
            num_classes=train_ds.NUM_CLASSES, **cfg.model.args
        ).to(device)
        self.opt = opt_cls(self.model.parameters(), **cfg.opt.args)
        self.device = device
        self.train_ds = train_ds
        self.train_loader = torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.cfg.train_batch_size,
            shuffle=True, collate_fn=self.collate_fn
        )
        self.val_ds = val_ds
        self.val_loader = torch.utils.data.DataLoader(
            self.val_ds, batch_size=self.cfg.val_batch_size,
            collate_fn=self.collate_fn
        )

        self._num_iter = 0
    
    def collate_fn(self, objects):
        wavs, labels = list(zip(*objects))
        wavs = self.model.collate(wavs)
        labels = torch.stack([self.model.align(wav, label)
                              for wav, label in zip(wavs, labels)], dim=0)
        return wavs, labels
    
    def load_state_dict(self, sd):
        self.model.load_state_dict(sd['model'])

        opt_sd = self.opt.state_dict()
        opt_sd['state'] = sd['opt']['state']
        self.opt.load_state_dict(opt_sd)

        self._num_iter = sd['num_iter']

    def state_dict(self):
        sd = OrderedDict()
        sd['model'] = self.model.state_dict()
        sd['opt'] = self.opt.state_dict()
        sd['num_iter'] = self._num_iter
        return sd

    def train_loop(self, num_epochs, checkpoint_dir, checkpoint_freq):
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, num_epochs + 1):
            print(f'Epoch {i}:')
            self.train_epoch()
            self.validation()
            print('-' * 100)
            if checkpoint_dir is not None and i % checkpoint_freq == 0:
                checkpoint_path = checkpoint_dir / f'{self._num_iter}.pth'
                print(f'Saving checkpoint to {checkpoint_path}')
                torch.save(self.state_dict(), checkpoint_path)
                wandb.save(str(checkpoint_path))

    def train_epoch(self):
        self.model.train()
        total_loss = torch.zeros(1, device=self.device)
        total_acc = torch.zeros(1, device=self.device)
        for x, y in tqdm(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            output = self.model(x)
            loss = F.cross_entropy(output.reshape(-1, output.shape[-1]),
                                   y.flatten())
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            total_loss += loss.detach()
            acc = (output.argmax(dim=-1) == y).float().mean()
            total_acc += acc

            data = {
                'train/loss': loss.detach(),
                'train/accuracy': acc,
                'train/lr': self.opt.param_groups[0]['lr']
            }
            wandb.log(
                data=data,
                step=self._num_iter,
                commit=(self._num_iter % 10 == 0)
            )
            self._num_iter += 1
        print('Train loss:', total_loss.item() / len(self.train_loader))
        print('Train accuracy:', total_acc.item() / len(self.train_loader))

    @torch.no_grad()
    def validation(self):
        self.model.eval()
        total_loss = torch.zeros(1, device=self.device)
        total_acc = torch.zeros(1, device=self.device)
        total_f1_scores = torch.zeros(self.val_ds.NUM_CLASSES,
                                      device=self.device)
        table = wandb.Table(columns=['wav', 'prediction'])
        for x, y in tqdm(self.val_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            output = self.model(x)
            loss = F.cross_entropy(output.reshape(-1, output.shape[-1]),
                                   y.flatten())
            total_loss += loss

            pred = output.argmax(dim=-1)
            total_acc += (pred == y).float().mean()
            total_f1_scores += f1_score(pred, y, len(total_f1_scores))

            wav = x[0].cpu().numpy()
            prediction_img = visualize(
                wav, pred[0].cpu().numpy(),
                y[0].cpu().numpy(), self.val_ds.IND2LABEL
            )
            table.add_data(
                wandb.Audio(wav, sample_rate=self.model.INPUT_SR),
                wandb.Image(prediction_img)
            )

        val_loss = total_loss.item() / len(self.val_loader)
        val_acc = total_acc.item() / len(self.val_loader)
        f1_scores = total_f1_scores.cpu().numpy() / len(self.val_loader)

        data = {
            'val/loss': val_loss,
            'val/accuracy': val_acc,
            'val/table': table
        }
        for ind, score in enumerate(f1_scores):
            data[f'val/{self.val_ds.IND2LABEL[ind]}_f1_score'] = score.item()
        wandb.log(
            data=data,
            step=self._num_iter
        )
        print('Validation loss:', val_loss)
        print('Validation accuracy:', val_acc)
