import ipdb
from pprint import pprint
import os
from argparse import ArgumentParser, Namespace
import datetime
from dateutil import tz
import random
import numpy as np
import torch
import warnings
from lightning import seed_everything, Trainer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from melp.datasets.pretrain_datamodule import SleepDataModule
from melp.models.uniclip_model import UniCLIPModel
from melp.models.simclr_model import SimCLRModel
from melp.models.mae_model import MAEModel
from melp.models.dino_model import DINOModel
from wav2sleep.config import *
from train_config import *

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def param_stats(model: torch.nn.Module, verbose: bool = False):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"{'Name':40s} {'Shape':20s} {'#Params':>10s} {'Train?':>6s}")
        print("-" * 80)
        for name, p in model.named_parameters():
            print(f"{name:40s} {str(list(p.shape)):20s} {p.numel():10d} {str(p.requires_grad):>6s}")
        print("-" * 80)
    print(f"Total parameters:     {total / 1e6:.3f} M ({total})")
    print(f"  Trainable params:   {trainable / 1e6:.3f} M ({trainable})")
    print(f"  Frozen params:      {(total-trainable) / 1e6:.3f} M ({total-trainable})")
def main(hparams: Namespace):

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    extension = f"sleep_unimodal_{hparams.model_name}_{extension}"
    ckpt_dir = os.path.join(
        CKPT_PATH, f"logs/melp/ckpts/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    if hparams.model_name in MODEL_LIST:
        callbacks = [
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(monitor="train/loss", dirpath=ckpt_dir,
                            save_last=True, every_n_train_steps=500, mode="max", save_top_k=2,
                            auto_insert_metric_name=True),
            EarlyStopping(monitor="train/loss", min_delta=0,
                        patience=5, verbose=True, mode="max"),
        ]
    else:
        raise NotImplementedError
    logger_dir = os.path.join(CKPT_PATH, "logs/melp")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project=hparams.wandb_proj_name, save_dir=logger_dir, name=extension)
    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        accelerator="gpu",
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        devices=hparams.num_devices,
        num_nodes=hparams.num_nodes,
        strategy="ddp_find_unused_parameters_true",
        callbacks=callbacks,
        logger=wandb_logger
    )

    # ------------------------
    # 2 INIT LIGHTNING MODEL and lightning datamodule
    # ------------------------
    hparams.exp_log_dir = os.path.join(
        CKPT_PATH, f"data/{extension}/exp_logs")
    dm = SleepDataModule(
            is_pretrain    = 1, # fixed in pre-training
            csv_dir        = SPLIT_DATA_FOLDER,
            train_edf_cols = TRAIN_EDF_COLS,  
            batch_size     = hparams.batch_size,
            num_workers    = hparams.num_workers,
            window_size = 10 * 30,
        )
    if hparams.model_name == "clip":
        model = UniCLIPModel(**vars(hparams))
    elif hparams.model_name == "simclr":
        model = SimCLRModel(**vars(hparams))
    elif hparams.model_name == "mae":
        model = MAEModel(**vars(hparams))
    elif hparams.model_name == "dino":
        model = DINOModel(**vars(hparams))
    else:
        raise NotImplementedError

    # model.training_steps_per_epoch = len(dm.train_dataloader()) // hparams.accumulate_grad_batches // hparams.num_devices
    pprint(vars(hparams))

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    # tuner = Tuner(trainer)
    # Find optimal batch size
    # optimal_batch_size = tuner.scale_batch_size(model=model, datamodule=datamodule, init_val=128,
    #                                             mode="binsearch")
    # datamodule.batch_size = optimal_batch_size
    # print(f"Optimal batch size: {optimal_batch_size}")
    # Find optimal learning rate
    # lr_finder = tuner.lr_find(model=model, datamodule=datamodule, max_lr=1e-3)
    # model.lr = lr_finder.suggestion()
    
    param_stats(model, verbose=True)
    trainer.fit(model, datamodule = dm)


if __name__ == '__main__':
    parser = ArgumentParser(description="Pretraining Multimodal ECG Foundation Model.")
    parser.add_argument("--model_name", type=str, default="clip",
                        choices=MODEL_LIST)
    
    parser.add_argument("--psg_encoder_name", type=str, default="resnet101")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_data_pct", type=float, default=1.)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--text_encoder_name", type=str, default="google/flan-t5-base")
    parser.add_argument("--clip_loss_weight", type=float, default=1.)
    parser.add_argument("--wandb_proj_name", type=str, default="melp")
    # parser.add_argument("--caption_loss_weight", type=float, default=1.)
    # parser.add_argument("--local_loss_weight", type=float, default=1.)
    # parser.add_argument("--model_size", type=str, default="base")

    

    hparams = parser.parse_args()

    # set random seed
    random.seed(hparams.seed)
    np.random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    seed_everything(hparams.seed)
    main(hparams)