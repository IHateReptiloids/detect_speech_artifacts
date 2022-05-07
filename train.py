import hydra
from omegaconf import DictConfig
import torch
import wandb

from src.datasets import ConcatDataset, LibriStutter, SSPNetVC
from src.models import BCResNet, CRNN, Wav2Vec2Pretrained
from src.schedulers import CosineAnnealingWarmupScheduler, IdScheduler
from src.trainers import FramewiseClassificationTrainer

DATASETS = {
    'libri_stutter': LibriStutter,
    'ssp_net_vc': SSPNetVC
}

MODELS = {
    'bc_resnet': BCResNet,
    'crnn': CRNN,
    'wav2vec2': Wav2Vec2Pretrained
}

OPTS = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}

SCHEDULERS = {
    'cosine_annealing_warmup': CosineAnnealingWarmupScheduler,
    'id_scheduler': IdScheduler
}

TRAINERS = {
    'framewise_clsf_trainer': FramewiseClassificationTrainer
}


def make_dataset(ds_cfg, target_sr):
    datasets = []
    for name, args in ds_cfg.items():
        datasets.append(DATASETS[name](target_sr=target_sr, **args))
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


@hydra.main(config_path='configs', config_name='config')
def main(cfg: DictConfig):
    wandb.init(config=cfg, group=cfg.wandb_group)

    torch.manual_seed(cfg.seed)

    train_ds = make_dataset(cfg.train_ds, MODELS[cfg.model.name].INPUT_SR)
    val_ds = make_dataset(cfg.val_ds, MODELS[cfg.model.name].INPUT_SR)
    if train_ds.NUM_CLASSES != val_ds.NUM_CLASSES:
        raise ValueError('Train dataset and validation dataset have ' +\
                         'different number of classes')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    trainer = FramewiseClassificationTrainer(
        cfg, MODELS[cfg.model.name], OPTS[cfg.opt.name],
        SCHEDULERS[cfg.scheduler.name], device, train_ds, val_ds
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
