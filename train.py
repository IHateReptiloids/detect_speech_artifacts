import hydra
from omegaconf import DictConfig
import torch
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

TRAINERS = {
    'unsupervised_ft_trainer': UnsupervisedFineTuningTrainer
}


@hydra.main(config_path='configs', config_name='config')
def main(cfg: DictConfig):
    wandb.init(config=cfg)

    torch.manual_seed(cfg.seed)

    train_ds = DATASETS[cfg.train_ds.name](
        target_sr=MODELS[cfg.model.name].INPUT_SR, **cfg.train_ds.args
    )
    val_ds = DATASETS[cfg.val_ds.name](
        target_sr=MODELS[cfg.model.name].INPUT_SR, **cfg.val_ds.args
    )
    if train_ds.NUM_CLASSES != val_ds.NUM_CLASSES:
        raise ValueError('Train dataset and validation dataset have ' +\
                         'different number of classes')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    trainer = UnsupervisedFineTuningTrainer(
        cfg, MODELS[cfg.model.name], OPTS[cfg.opt.name],
        device, train_ds, val_ds
    )

    wandb.watch((trainer.model,), log='all', log_freq=cfg.wandb_log_freq)
    if cfg.wandb_file_name is not None and cfg.wandb_run_path is not None:
        f = wandb.restore(cfg.wandb_file_name, cfg.wandb_run_path,
                          cfg.checkpoint_dir)
        trainer.load_state_dict(torch.load(f.name, map_location=device))
        f.close()
    trainer.train_loop(cfg.num_epochs, cfg.checkpoint_dir, cfg.checkpoint_freq)


if __name__ == '__main__':
    main()
