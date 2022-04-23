import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import wandb

from src.datasets import SSPNetVC
from src.models import Wav2Vec2Pretrained
from src.trainers import UnsupervisedFineTuningTrainer

DATASETS = {
    'ssp_net_vc': SSPNetVC
}

MODELS = {
    'wav2vec2': Wav2Vec2Pretrained
}

OPTS = {
    'adam': torch.optim.Adam
}


@hydra.main(config_path='configs', config_name='config')
def main(cfg: DictConfig):
    wandb.init()

    torch.manual_seed(cfg.seed)

    train_ds = DATASETS[cfg.train_ds.name](**cfg.train_ds.args)
    val_ds = DATASETS[cfg.val_ds.name](**cfg.val_ds.args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = MODELS[cfg.model.name](**cfg.model.args)

    for p in model.parameters():
        p.requires_grad_(False)
    model = nn.Sequential(
        model,
        nn.Linear(model.num_features, 3)
    )

    opt = OPTS[cfg.opt.name](model[1].parameters(), **cfg.opt.args)
    trainer = UnsupervisedFineTuningTrainer(model, opt, device)

    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=cfg.batch_size,
                                               shuffle=True,
                                               collate_fn=trainer.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size,
                                             collate_fn=trainer.collate_fn)

    wandb.watch((model,), log='all', log_freq=cfg.wandb_log_freq)
    if cfg.wandb_file_name is not None and cfg.wandb_run_path is not None:
        f = wandb.restore(cfg.wandb_file_name, cfg.wandb_run_path,
                          cfg.checkpoint_dir)
        trainer.load_state_dict(torch.load(f.name, map_location=device))
        f.close()
    trainer.train_loop(cfg.num_epochs, train_loader, val_loader,
                       cfg.checkpoint_dir, cfg.checkpoint_freq)


if __name__ == '__main__':
    main()

