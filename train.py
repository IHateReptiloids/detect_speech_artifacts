import torch
import torch.nn as nn
import wandb

from src.datasets import SSPNetVC
from src.models import Wav2Vec2Pretrained
from src.trainers import UnsupervisedFineTuningTrainer


wandb.init()

seed = 8228
torch.manual_seed(8228)

train_ds = SSPNetVC(data_path='data/ssp/data',
                    labels_path='data/ssp/train_labels.txt',
                    target_sr=16_000)
val_ds = SSPNetVC(data_path='data/ssp/data',
                  labels_path='data/ssp/val_labels.txt',
                  target_sr=16_000)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = Wav2Vec2Pretrained('base')
for p in model.parameters():
    p.requires_grad_(False)
model = nn.Sequential(
    model,
    nn.Linear(model.num_features, 3)
)
opt = torch.optim.Adam(model[1].parameters(), lr=3e-4)
trainer = UnsupervisedFineTuningTrainer(model, opt, device)

batch_size = 16
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                           shuffle=True,
                                           collate_fn=trainer.collate_fn)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,
                                         collate_fn=trainer.collate_fn)

wandb.watch((model,), log='all', log_freq=100)
trainer.train_loop(10, train_loader, val_loader, 'checkpoints/wav2vec2-ssp-net-vc')
