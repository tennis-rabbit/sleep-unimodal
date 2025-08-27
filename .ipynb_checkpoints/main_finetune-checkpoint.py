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
from melp.models.ssl_finetuner import SSLFineTuner

from train_config import *

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

'''
CUDA_VISIBLE_DEVICES=0 python main_finetune.py \
    --model_name melp --dataset_name icbeb \
    --train_data_pct 0.01 \
    --ckpt_path CKPT_PATH \
    --num_devices 1
'''
def main(hparams: Namespace):

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    extension = f"melp_finetune_{hparams.model_name}_{extension}"
    ckpt_dir = os.path.join(
        CKPT_PATH, f"logs/melp_finetune/ckpts/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_auc", dirpath=ckpt_dir,
                        save_last=False, mode="max", save_top_k=1,
                        auto_insert_metric_name=True),
        EarlyStopping(monitor="val_auc", min_delta=0,
                      patience=5, verbose=False, mode="max"),
    ]
    logger_dir = os.path.join(CKPT_PATH, "logs/melp_finetune")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="melp_finetune", save_dir=logger_dir, name=extension)
    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        accelerator="gpu",
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        deterministic=True,
        devices=hparams.num_devices,
        strategy="ddp_find_unused_parameters_true",
        precision="bf16-mixed",
        callbacks=callbacks,
        logger=wandb_logger
    )

    # ------------------------
    # 2 INIT LIGHTNING MODEL and lightning datamodule
    # ------------------------
    hparams.exp_log_dir = os.path.join(
        CKPT_PATH, f"data/{extension}/exp_logs")
    datamodule = SleepDataModule(
            is_pretrain    = 0, # fixed in pre-training
            data_pct       = hparams.train_data_pct,
            csv_dir        = SPLIT_DATA_FOLDER,
            train_edf_cols = TRAIN_EDF_COLS,   # 你的通道列表
            event_cols     = hparams.eval_label,
            batch_size     = hparams.batch_size,
            num_workers    = hparams.num_workers,
            window_size = 10 * 30,
        )
    hparams.num_classes = datamodule.train_dataloader().dataset.num_classes
    print(hparams.num_classes)
    # hparams.num_patches = datamodule.train_dataloader().dataset.patches_per_lead * 12
    hparams.training_steps_per_epoch = len(datamodule.train_dataloader()) // hparams.accumulate_grad_batches // hparams.num_devices
    if hparams.model_name == "clip":
        pretrain_model = UniCLIPModel.load_from_checkpoint(hparams.ckpt_path)
    else:
        raise NotImplementedError
    pprint(vars(hparams))

    model = SSLFineTuner(backbones=pretrain_model.encoders, use_which_backbone = 'ecg', **vars(hparams)) # could use args to assign which encoder


    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == '__main__':
    parser = ArgumentParser(description="Pretraining Multimodal ECG Foundation Model.")
    parser.add_argument("--model_name", type=str, default="clip",
                        )
    parser.add_argument("--dataset_name", type=str, default="ptbxl_super_class",
                        choices=["ptbxl_super_class", "ptbxl_sub_class", "ptbxl_form", "ptbxl_rhythm",
                                 "icbeb", "chapman"])
    parser.add_argument("--eval_label", type=str, default="Stage",
                        )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_data_pct", type=float, default=1.)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--in_features", type=int, default=256)
    parser.add_argument("--use_ecg_patch", action="store_true")
    hparams = parser.parse_args()

    # set random seed
    random.seed(hparams.seed)
    np.random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    seed_everything(hparams.seed)
    main(hparams)