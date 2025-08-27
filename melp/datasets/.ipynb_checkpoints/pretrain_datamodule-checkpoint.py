import os
from typing import List
import ipdb
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import pandas as pd
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from melp.paths import SPLIT_DIR
from pathlib import Path
from typing import List, Sequence, Optional, Dict, Union

import pandas as pd          
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from melp.datasets.pretrain_dataset import SleepEpochDataset    

class SleepDataModule(LightningDataModule):

    def __init__(
        self,
        csv_dir: str | Path,
        *,
        is_pretrain,
        data_pct = 1,
        val_dataset_list: Optional[List[str]] = None,
        batch_size: int = 128,
        num_workers: int = 4,
        patient_cols: Optional[Union[str, Sequence[str]]] = None,
        event_cols: Optional[Union[str, Sequence[str]]] = None,
        train_edf_cols: Sequence[str] | None,  # 传给 Dataset
        transforms=None,
        n_views: int = 1,
        cache_size: int = 8,                   # 透传给 Dataset
        sample_rate: int = 128,
        window_size: int = 30,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["transforms"])  

        self.csv_dir   = csv_dir
        self.transforms = transforms
        self.n_views    = n_views
        self.pin_memory = pin_memory
        self.is_pretrain = is_pretrain
        self.patient_cols = patient_cols
        self.event_cols = event_cols
        self.data_pct = data_pct


    # ---------- 3. DataLoader ----------
    def train_dataloader(self):
        if self.is_pretrain == 1:
            train_set = SleepEpochDataset(
                    csv_dir       = self.csv_dir,
                    split         = "pretrain",
                    data_pct      = 1,
                    train_edf_cols= self.hparams.train_edf_cols,
                    transform     = self.transforms,
                    sample_rate   = self.hparams.sample_rate,
                    window_size   = self.hparams.window_size,
                    cache_size    = self.hparams.cache_size,
                )
        else:
            train_set = SleepEpochDataset(
                    csv_dir       = self.csv_dir,
                    split         = "train",
                    data_pct      = self.data_pct,
                    patient_cols  = self.patient_cols,
                    event_cols    = self.event_cols,
                    train_edf_cols= self.hparams.train_edf_cols,
                    transform     = self.transforms,
                    sample_rate   = self.hparams.sample_rate,
                    window_size   = self.hparams.window_size,
                    cache_size    = self.hparams.cache_size,
                )
        return DataLoader(
            train_set,
            batch_size     = self.hparams.batch_size,
            shuffle        = True,
            num_workers    = self.hparams.num_workers,
            pin_memory     = self.pin_memory,
            persistent_workers = self.hparams.num_workers > 0,
        )

    def val_dataloader(self):
        if self.hparams.val_dataset_list:       
            # we could be flexible here. for default setting, we just use val split of pre-training data
            val_sets = [
                    SleepEpochDataset(
                        csv_dir       = self.csv_dir,
                        split         = "val",
                        data_pct      = self.data_pct,
                        patient_cols   = self.patient_cols,
                        event_cols   = self.event_cols,
                        train_edf_cols= self.hparams.train_edf_cols,
                        transform     = None,        
                        sample_rate   = self.hparams.sample_rate,
                        window_size   = self.hparams.window_size,
                        cache_size    = self.hparams.cache_size,
                    )
                    for _ in self.hparams.val_dataset_list
                ]
        else:
            # we could be flexible here. for default setting, we just use val split of pre-training data
            val_sets = [
                    SleepEpochDataset(
                        csv_dir       = self.csv_dir,
                        split         = "val",
                        data_pct      = self.data_pct,
                        patient_cols   = self.patient_cols,
                        event_cols   = self.event_cols,
                        train_edf_cols= self.hparams.train_edf_cols,
                        transform     = None,
                        sample_rate   = self.hparams.sample_rate,
                        window_size   = self.hparams.window_size,
                        cache_size    = self.hparams.cache_size,
                    )
                ]
        return [
            DataLoader(
                ds,
                batch_size     = self.hparams.batch_size,
                shuffle        = False,
                num_workers    = self.hparams.num_workers,
                pin_memory     = self.pin_memory,
                persistent_workers = self.hparams.num_workers > 0,
            )
            for ds in val_sets
        ]

    def test_dataloader(self):
        test_set = SleepEpochDataset(
                csv_dir       = self.csv_dir,
                split         = "test",
                patient_cols   = self.patient_cols,
            event_cols   = self.event_cols,
                train_edf_cols= self.hparams.train_edf_cols,
                transform     = None,
                sample_rate   = self.hparams.sample_rate,
                window_size   = self.hparams.window_size,
                cache_size    = self.hparams.cache_size,
            )
        return DataLoader(
            test_set,
            batch_size     = self.hparams.batch_size,
            shuffle        = False,
            num_workers    = self.hparams.num_workers,
            pin_memory     = self.pin_memory,
            persistent_workers = self.hparams.num_workers > 0,
        )

