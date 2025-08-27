'''
The script contains pretraining dataset for MIMIC-IV-ECG
'''
import ipdb
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import List
import numpy as np
from contextlib import suppress
from tqdm import tqdm
from einops import rearrange
import wfdb
import itertools
from melp.paths import SPLIT_DIR
from melp.datasets.augmentations import RandomLeadsMask


class ECG_Text_Dataset(Dataset):
    """ Dataset for MIMIC-IV-ECG"""
    def __init__(self, 
                 split: str, 
                 dataset_dir: str, 
                 dataset_list: List = ["mimic-iv-ecg"], 
                 data_pct: float = 1, 
                 transforms = None,
                 n_views: int = 1,
                 use_cmsc: bool = False,
                 use_rlm: bool = False,
                 num_beats: int = 1,
                 ):
        
        super().__init__()
        
        self.split = split
        self.dataset_dir = dataset_dir
        self.dataset_list = dataset_list
        self.data_pct = data_pct
        self.use_cmsc = use_cmsc
        self.use_rlm = use_rlm
        self.n_views = n_views
        if transforms is None:
            self.augs = []
        else:
            self.augs = transforms
        if self.use_rlm:
            # random mask 50% leads for each samplesa
            self.augs.append(
                RandomLeadsMask(p=1, mask_leads_selection="random", mask_leads_prob=0.5)
                )

        all_df = []
        for dataset_name in self.dataset_list:
            df = pd.read_csv(SPLIT_DIR / f"{dataset_name}/{self.split}.csv", low_memory=False)
            df["path"] = df["path"].apply(lambda x: os.path.join(self.dataset_dir, dataset_name, x))
            print(f"Loading {dataset_name} {self.split} dataset: total {len(df)} samples")
            all_df.append(df)
        self.df = pd.concat(all_df)
        # sample data
        self.df = self.df.sample(frac=self.data_pct).reset_index(drop=True)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = torch.tensor([row["subject_id"]]).long()
        report = row["total_report"]
        # ecg = np.load(row["path"])
        ecg = wfdb.rdsamp(row["path"])[0].T
        # normalize ecg into 0 - 1
        ecg = (ecg - np.min(ecg)) / (np.max(ecg) - np.min(ecg) + 1e-8)
        ecg = torch.tensor(ecg).float()
        num_leads = ecg.shape[0]

        if self.use_cmsc:
            num_samples = ecg.size(1)
            ecg1 = ecg[:, :num_samples//2]
            ecg2 = ecg[:, num_samples//2:]
            for aug in self.augs:
                ecg1 = aug(ecg1)
                ecg2 = aug(ecg2)
            ecg = torch.stack([ecg1, ecg2], dim=0)
            patient_id = torch.cat([patient_id, patient_id], dim=0)
        else:
            if self.n_views == 1:
                for aug in self.augs:
                    ecg = aug(ecg)
            else:
                ecg_list = []
                for _ in range(self.n_views):
                    # original ecg
                    ecg_ = ecg.clone()
                    for aug in self.augs:
                        ecg_ = aug(ecg_)
                    ecg_list.append(ecg_)
                ecg = torch.stack(ecg_list, dim=0)
                patient_id = torch.cat([patient_id]*self.n_views, dim=0)

        return {
            "id": row["id"],
            "patient_id": patient_id,
            "ecg": ecg,
            "report": report
        }

# def custom_collate_fn(batch):
#     ids = [x["id"] for x in batch]

#     ecgs = torch.stack([x["ecg"] for x in batch], dim=0)
#     # if batch[0]["ecg"].ndim == 2:
#     #     # If not use CMSE, then stack the ecg
#     #     ecgs = torch.stack([x["ecg"] for x in batch])
#     # elif batch[0]["ecg"].ndim == 3:
#     #     # If use CMSE, then concatenate the ecg
#     #     ecgs = torch.cat([x["ecg"] for x in batch], dim=0)
#     # else:
#     #     raise ValueError("Invalid ECG dimension")
    
    # reports = [x["report"] for x in batch]
    # mask_reports = [x["mask_report"] for x in batch]
#     reports = [x["report"] for x in batch]
    
#     patient_ids = torch.stack([x["patient_id"] for x in batch], dim=0)

    # return {
    #     "id": ids,
    #     "ecg": ecgs,
    #     "report": reports,
    #     "mask_report": mask_reports,
    #     "patient_ids": patient_ids
    # }
#     return {
#         "id": ids,
#         "ecg": ecgs,
#         "report": reports,
#         "patient_ids": patient_ids
#     }




# dataset.py
import numpy as np
import pandas as pd
import gzip, pickle, xxhash 

from pathlib import Path
from functools import lru_cache
from typing import Sequence, Optional, Tuple, Union, List, Dict

import time 
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from wav2sleep.config import *
from wav2sleep.data.utils import interpolate_index
# def read_csv_fast(path, *, usecols=None, dtypes=None, nrows=None):
    
#     return pd.read_csv(
#         path,
#         engine="pyarrow",
#         usecols=usecols,
#         dtype=dtypes,                 
#         nrows=nrows,                 
#         dtype_backend="pyarrow",     
#     )
# class SleepEpochDataset(Dataset):
#     """
#     A PyTorch Dataset that delivers 30-second sleep epochs together
#     with optional patient-level labels.

#     Parameters
#     ----------
#     epoch_df : pd.DataFrame
#         Epoch-level table. Must contain at least:
#             - "nsrrid"       : patient ID
#             - "epoch_id"     : 1-based epoch index
#             - "path_head"    : prefix to the signal file on disk
#     patient_df : pd.DataFrame | str | Path
#         Patient-level table or the CSV file path; must contain
#         "nsrrid" plus any downstream label columns.
#     split : {"train", "val", "test"}
#         Which split to use. Partitioning is done by patient ID
#         so all epochs from the same patient stay in the same split.
#     target_cols : str | Sequence[str] | None
#         The patient-level label column(s) to return.  None = no labels
#         (e.g. for self-supervised pre-training).
#     test_size, val_size : float
#         Fractions for patient-level train/val/test split.
#     random_state : int
#         Seed for deterministic splitting.
#     sample_rate : int
#         Sampling rate (Hz) of the pre-processed signal.
#     cache_size : int
#         LRU cache size for whole-night signals to avoid reloading.
#     transform : callable | None
#         Optional transform / augmentation applied to the epoch tensor.
#     split_ids : dict | None
#         Pre-defined {"train": [...], "val": [...], "test": [...]} lists.
#         If supplied, overrides the random split.
#     """

#     def __init__(
#         self,
#         csv_dir = '/scratch/besp/shared_data/sleep_data_split_test',
#         split: str = "train",
#         *,
#         target_cols: Optional[Union[str, Sequence[str]]] = None,
#         train_edf_cols = None,
#         test_size: float = 0.15,
#         val_size: float = 0.15,
#         random_state: int = 1337,
#         sample_rate: int = 128,
#         window_size: int = 30,
#         cache_size: int = 8,
#         transform=None,
#         split_ids: Optional[Dict[str, Sequence[str]]] = None,
#     ):
#         assert split in {"pretrain", "train", "val", "test"}
#         self.transform = transform
#         self.sample_rate = sample_rate
#         self.window_size = window_size
        
        
        
#         #######################################################
#         epoch_usecols   = ["nsrrid", "epoch_id", "path_head"]
#         patient_usecols = ["nsrrid"]
#         if target_cols:
#             if isinstance(target_cols, str):
#                 target_cols = [target_cols]
#             patient_usecols += list(target_cols)

        
#         epoch_dtypes = {
#             "nsrrid": "string[pyarrow]",
#             "epoch_id": "int32",
#             "path_head": "string[pyarrow]",
#         }
#         patient_dtypes = {"nsrrid": "string[pyarrow]"}
#         if target_cols:
#             for c in target_cols:
#                 patient_dtypes[c] = "string[pyarrow]"  

#         if split == "pretrain":
#             patient_df = read_csv_fast(f"{csv_dir}/pretrain_patients.csv",
#                                        usecols=patient_usecols, dtypes=patient_dtypes)
#             epoch_df   = read_csv_fast(f"{csv_dir}/pretrain_epochs.csv",
#                                        usecols=epoch_usecols, dtypes=epoch_dtypes)
#         elif split == "train":
#             patient_df = read_csv_fast(f"{csv_dir}/val_patients.csv",
#                                        usecols=patient_usecols, dtypes=patient_dtypes)
#             epoch_df   = read_csv_fast(f"{csv_dir}/val_epochs.csv",
#                                        usecols=epoch_usecols, dtypes=epoch_dtypes)
#         elif split == "val":
#             patient_df = read_csv_fast(f"{csv_dir}/val_patients.csv",
#                                        usecols=patient_usecols, dtypes=patient_dtypes)
#             epoch_df   = read_csv_fast(f"{csv_dir}/val_epochs.csv",
#                                        usecols=epoch_usecols, dtypes=epoch_dtypes)
#         elif split == "test":
#             patient_df = read_csv_fast(f"{csv_dir}/test_patients.csv",
#                                        usecols=patient_usecols, dtypes=patient_dtypes)
#             epoch_df   = read_csv_fast(f"{csv_dir}/test_epochs.csv",
#                                        usecols=epoch_usecols, dtypes=epoch_dtypes)

#         self.epoch_df   = epoch_df
#         self.patient_df = patient_df.set_index("nsrrid")
#         self.train_edf_cols = train_edf_cols
#         self.target_cols = target_cols
#         #######################################################
#     # ───────── Dataset API ─────────

#     def __len__(self) -> int:
#         return len(self.epoch_df)

#     def __getitem__(self, idx: int):
        
#         row = self.epoch_df.iloc[idx]
#         nsrrid = row["nsrrid"]
#         epoch_id = int(row["epoch_id"])

#         # 1) load full-night signal and slice to this epoch
#         channel_indicator = pd.read_csv(row['path_head'] + '_channel_indicator.csv')

#         dfs = []
#         for col_name in self.train_edf_cols:

#             if channel_indicator.loc[0, col_name] == 0:
#                 real_index = pd.Index(np.arange(temp_sig.index.min(), np.ceil(temp_sig.index.max()), 1 / TARGET_RATE), name="sec") # incase: 810.0-> 839.9921875
#                 sig_df = pd.DataFrame(
#                     np.zeros(real_index.size, dtype=float),
#                     index=real_index,
#                     columns=[col_name]          
#                 )
#             else:
#                 # overall: 8 batchsize -> 200G - 8 num_of_workers: 0.05~0.1s
#                 # t00 = time.perf_counter() 
#                 temp_sig = self._load_patient_array(nsrrid, row["path_head"], col_name = col_name, epoch_id = epoch_id)

#                 ##############################################################
#                 # 0.003 ~ 0.005 sec
#                 # t01 = time.perf_counter() 
#                 # resample to a shared frequency
#                 TARGET_RATE = self.sample_rate                    # e.g. 128 Hz
#                 period_ms   = 1000 / TARGET_RATE                  # 7.812 ms
#                 samples_per_epoch = self.window_size * TARGET_RATE              # 3840
                
#                 real_index = pd.Index(np.arange(temp_sig.index.min(), np.ceil(temp_sig.index.max()), 1 / TARGET_RATE)) # incase: 810.0-> 839.9921875
                
#                 sig_df = interpolate_index(temp_sig, real_index,
#                                   method="linear", squeeze=False)
#                 # t02 = time.perf_counter() 
#                 # print(f"{col_name} load: {t01 - t00:.3f} s")
#                 # print(f"{col_name} resample: {t02 - t01:.3f} s")
#             ##############################################################
            
            
#             dfs.append(sig_df)
        
#         full_sig = pd.concat(dfs, axis = 1)
        
#         x = torch.tensor(full_sig.values, dtype=torch.float32)
        
#         # 1.5) can add other transformation here
#         if self.transform:
#             x = self.transform(x)
#         # transpose to (C, T)
#         x = x.transpose(0, 1)
#         # 2) add patient-level label(s) if requested
#         if self.target_cols:
#             y = torch.tensor(
#                 self.patient_df.loc[nsrrid, self.target_cols].values.astype(float),
#                 dtype=torch.float32,
#             )
#             data = {
#                 'psg': x,
#                 'label': y,
#             }
#             return data
#         else:
#             data = {
#                 'psg': x,
#             }
#             return data


#     def _build_signal_path(self, path_head: str, col_name = 'ECG', epoch_id = 0) -> Path:
#         """
#         Convert path_head to the actual signal file path.
#         Default: '<path_head>_data.npz' with key 'signal'.
#         Adjust if your filenames / keys differ.
#         """
        
#         return Path(path_head + f"/epoch-{epoch_id:05d}_{col_name}.npz")

#     def _load_patient_array(self, nsrrid: str, path_head: str, col_name = 'ECG', epoch_id = 0) -> np.ndarray:
#         """
#         Load the full-night signal into a NumPy array.
#         If you have multiple channels, return shape (C, T).
#         """
        
#         fp = self._build_signal_path(path_head, col_name, epoch_id)
#         if not fp.is_file():
#             raise FileNotFoundError(f"Signal file missing: {fp}")
#         with np.load(fp, allow_pickle=True) as npz:
            
#             data = npz['values']
#             index = npz['index']

#             df_stg = pd.DataFrame(
#                 data,
#                 columns=[col_name]
#             )
#             df_stg.insert(0, "sec", index)
            
#             sig = df_stg.set_index("sec")           
            
#         return sig.astype(np.float32)

class SleepEpochDataset(Dataset):
    """
    A PyTorch Dataset that delivers 30-second sleep epochs together
    with optional patient-level labels.

    Parameters
    ----------
    epoch_df : pd.DataFrame
        Epoch-level table. Must contain at least:
            - "nsrrid"       : patient ID
            - "epoch_id"     : 1-based epoch index
            - "path_head"    : prefix to the signal file on disk
    patient_df : pd.DataFrame | str | Path
        Patient-level table or the CSV file path; must contain
        "nsrrid" plus any downstream label columns.
    split : {"train", "val", "test"}
        Which split to use. Partitioning is done by patient ID
        so all epochs from the same patient stay in the same split.
    target_cols : str | Sequence[str] | None
        The patient-level label column(s) to return.  None = no labels
        (e.g. for self-supervised pre-training).
    test_size, val_size : float
        Fractions for patient-level train/val/test split.
    random_state : int
        Seed for deterministic splitting.
    sample_rate : int
        Sampling rate (Hz) of the pre-processed signal.
    cache_size : int
        LRU cache size for whole-night signals to avoid reloading.
    transform : callable | None
        Optional transform / augmentation applied to the epoch tensor.
    split_ids : dict | None
        Pre-defined {"train": [...], "val": [...], "test": [...]} lists.
        If supplied, overrides the random split.
    """

    def __init__(
        self,
        csv_dir = '/scratch/besp/shared_data/five_min_sleep_data_split_test_with_cache',
        split: str = "train",
        *,
        patient_cols: Optional[Union[str, Sequence[str]]] = None,
        event_cols: Optional[Union[str, Sequence[str]]] = None,
        train_edf_cols = None,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 1337,
        sample_rate: int = 128,
        window_size: int = 30,
        cache_size: int = 8,
        transform=None,
        split_ids: Optional[Dict[str, Sequence[str]]] = None,
    ):
        assert split in {"pretrain", "train", "val", "test"}
        self.transform = transform
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.patient_cols = (
            [patient_cols] if isinstance(patient_cols, str) else patient_cols
        )

        # ───── patient-level DataFrame ─────
        
        if split == "pretrain":
            patient_df = pd.read_csv(csv_dir + '/pretrain_patients.csv')
            epoch_df = pd.read_csv(csv_dir + '/pretrain_epochs.csv')
            epoch_df = epoch_df[['nsrrid', 'seg_id', 'epoch_id', 'path_head']]
        elif split == "train":
            patient_df = pd.read_csv(csv_dir + '/val_patients.csv')
            epoch_df = pd.read_csv(csv_dir + '/val_epochs.csv')
        elif split == "val":
            patient_df = pd.read_csv(csv_dir + '/val_patients.csv')
            epoch_df = pd.read_csv(csv_dir + '/val_epochs.csv')
        elif split == "test":
            patient_df = pd.read_csv(csv_dir + '/test_patients.csv')
            epoch_df = pd.read_csv(csv_dir + '/test_epochs.csv')
        print(epoch_df.columns)
        self.split = split
        self.epoch_df = epoch_df
        if self.split == "pretrain":
            self.epoch_df = self.epoch_df.drop_duplicates(subset=["seg_id", "nsrrid"], keep="first").reset_index()
            print(f"{split} length: {len(self.epoch_df)} 5 min segments")
        self.patient_df = patient_df.set_index("nsrrid")

        
        # ───── LRU cache for whole-night signals ─────
        # self._load_patient_array = lru_cache(maxsize=cache_size)(
        #     self._load_patient_array
        # )
        self.train_edf_cols = train_edf_cols

    # ───────── Dataset API ─────────

    def __len__(self) -> int:
        return len(self.epoch_df)
    

    def _resample_df(self, df: pd.DataFrame, target_hz: int) -> pd.DataFrame:

        if not np.issubdtype(df.index.dtype, np.number):

            t = np.arange(len(df)) / float(target_hz)
            df = df.copy()
            df.index = t


        t0 = float(df.index.min())
        t1 = float(df.index.max())

        t_target = np.arange(t0, t0 + self.window_size, 1.0 / target_hz)

        if t_target[-1] > t1:
            t_target = t_target[t_target <= t1 + 1e-9]


        out = df.reindex(t_target).interpolate(method="linear", limit_direction="both")

        out = out.fillna(0.0)
        return out

    def __getitem__(self, idx: int):
        if self.split == "pretrain":
            row = self.epoch_df.iloc[idx]
            nsrrid = row["nsrrid"]
            epoch_id = int(row["seg_id"])


            cols = list(self.train_edf_cols) if self.train_edf_cols is not None else None
            df_epoch = self._load_epoch_all_df(row["path_head"], epoch_id, columns=cols)

            df_epoch = self._resample_df(df_epoch, self.sample_rate)
            if df_epoch.isna().any().any():
                raise ValueError(f"NaN detected in sample idx={idx}, nsrrid={nsrrid}, seg_id={epoch_id}")


            if cols is not None:
                for ch in cols:
                    if ch not in df_epoch.columns:
                        # print("missing channel", ch)
                        df_epoch[ch] = 0.0  
            df_epoch = df_epoch[cols]  

            samples_per_epoch = int(self.window_size * self.sample_rate)

            if len(df_epoch) < samples_per_epoch:
                pad = samples_per_epoch - len(df_epoch)
                tail = pd.DataFrame({c: 0.0 for c in df_epoch.columns},
                                    index=df_epoch.index[-1] + (np.arange(1, pad+1) / self.sample_rate))
                df_epoch = pd.concat([df_epoch, tail], axis=0)
            elif len(df_epoch) > samples_per_epoch:
                df_epoch = df_epoch.iloc[:samples_per_epoch]
            if df_epoch.isna().any().any():
                raise ValueError(f"NaN detected in sample idx={idx}, nsrrid={nsrrid}, seg_id={epoch_id}")

            # (T, C) -> tensor -> (C, T)
            # print(df_epoch.shape)
            x = torch.tensor(df_epoch.to_numpy(copy=False), dtype=torch.float32).t().contiguous()


            x = torch.clamp(x, min=-30, max=30)

            if self.patient_cols:
                y = torch.tensor(
                    self.patient_df.loc[nsrrid, self.patient_cols].values.astype(float),
                    dtype=torch.float32,
                )
                return {"psg": x, "label": y}
            elif self.event_cols:
                y = torch.tensor(
                    self.epoch_df.loc['epoch_id', self.event_cols].values.astype(float),
                    dtype=torch.float32,
                )
                return {"psg": x, "label": y}
            else:
                return {"psg": x}
        else:
            row = self.epoch_df.iloc[idx]
            nsrrid = row["nsrrid"]
            epoch_id = int(row["seg_id"])


            cols = list(self.train_edf_cols) if self.train_edf_cols is not None else None
            df_epoch = self._load_epoch_all_df(row["path_head"], epoch_id, columns=cols)

            df_epoch = self._resample_df(df_epoch, self.sample_rate)
            if df_epoch.isna().any().any():
                raise ValueError(f"NaN detected in sample idx={idx}, nsrrid={nsrrid}, seg_id={epoch_id}")


            if cols is not None:
                for ch in cols:
                    if ch not in df_epoch.columns:
                        # print("missing channel", ch)
                        df_epoch[ch] = 0.0  
            df_epoch = df_epoch[cols]  

            samples_per_epoch = int(self.window_size * self.sample_rate)

            if len(df_epoch) < samples_per_epoch:
                pad = samples_per_epoch - len(df_epoch)
                tail = pd.DataFrame({c: 0.0 for c in df_epoch.columns},
                                    index=df_epoch.index[-1] + (np.arange(1, pad+1) / self.sample_rate))
                df_epoch = pd.concat([df_epoch, tail], axis=0)
            elif len(df_epoch) > samples_per_epoch:
                df_epoch = df_epoch.iloc[:samples_per_epoch]
            if df_epoch.isna().any().any():
                raise ValueError(f"NaN detected in sample idx={idx}, nsrrid={nsrrid}, seg_id={epoch_id}")

            # (T, C) -> tensor -> (C, T)
            # print(df_epoch.shape)
            x = torch.tensor(df_epoch.to_numpy(copy=False), dtype=torch.float32).t().contiguous()


            x = torch.clamp(x, min=-30, max=30)

            if self.patient_cols:
                y = torch.tensor(
                    self.patient_df.loc[nsrrid, self.patient_cols].values.astype(float),
                    dtype=torch.float32,
                )
                return {"psg": x, "label": y}
            elif self.event_cols:
                y = torch.tensor(
                    self.epoch_df.loc['epoch_id', self.event_cols].values.astype(float),
                    dtype=torch.float32,
                )
                return {"psg": x, "label": y}
            else:
                return {"psg": x}



    def _build_epoch_all_path(self, path_head: str, epoch_id: int) -> Path:

        return Path(f"{path_head}/epoch-{epoch_id:05d}_all.parquet")


    def _load_epoch_all_df(self, path_head: str, epoch_id: int, columns=None) -> pd.DataFrame:

        fp = self._build_epoch_all_path(path_head, epoch_id)
        if not fp.is_file():
            raise FileNotFoundError(f"Parquet missing: {fp}")

        df = pd.read_parquet(fp)

        for c in df.columns:
            if not np.issubdtype(df[c].dtype, np.floating):
                with suppress(Exception):
                    df[c] = df[c].astype(np.float32)
        return df
# class FastSleepEpochDataset(Dataset):
#     """
#     A PyTorch Dataset that delivers 30-second sleep epochs together
#     with optional patient-level labels.

#     Parameters
#     ----------
#     epoch_df : pd.DataFrame
#         Epoch-level table. Must contain at least:
#             - "nsrrid"       : patient ID
#             - "epoch_id"     : 1-based epoch index
#             - "path_head"    : prefix to the signal file on disk
#     patient_df : pd.DataFrame | str | Path
#         Patient-level table or the CSV file path; must contain
#         "nsrrid" plus any downstream label columns.
#     split : {"train", "val", "test"}
#         Which split to use. Partitioning is done by patient ID
#         so all epochs from the same patient stay in the same split.
#     target_cols : str | Sequence[str] | None
#         The patient-level label column(s) to return.  None = no labels
#         (e.g. for self-supervised pre-training).
#     test_size, val_size : float
#         Fractions for patient-level train/val/test split.
#     random_state : int
#         Seed for deterministic splitting.
#     sample_rate : int
#         Sampling rate (Hz) of the pre-processed signal.
#     cache_size : int
#         LRU cache size for whole-night signals to avoid reloading.
#     transform : callable | None
#         Optional transform / augmentation applied to the epoch tensor.
#     split_ids : dict | None
#         Pre-defined {"train": [...], "val": [...], "test": [...]} lists.
#         If supplied, overrides the random split.
#     """

#     def __init__(
#         self,
#         csv_dir = '/scratch/besp/shared_data/sleep_data_split_test',
#         split: str = "train",
#         *,
#         target_cols: Optional[Union[str, Sequence[str]]] = None,
#         train_edf_cols = None,
#         test_size: float = 0.15,
#         val_size: float = 0.15,
#         random_state: int = 1337,
#         sample_rate: int = 128,
#         window_size: int = 30,
#         cache_size: int = 8,
#         transform=None,
#         split_ids: Optional[Dict[str, Sequence[str]]] = None,
#     ):
#         assert split in {"pretrain", "train", "val", "test"}
#         self.transform = transform
#         self.sample_rate = sample_rate
#         self.window_size = window_size
#         self.target_cols = (
#             [target_cols] if isinstance(target_cols, str) else target_cols
#         )

#         # ───── patient-level DataFrame ─────
        
#         if split == "pretrain":
#             patient_df = pd.read_csv(csv_dir + '/pretrain_patients.csv')
#             epoch_df = pd.read_csv(csv_dir + '/pretrain_epochs.csv')
#         elif split == "train":
#             patient_df = pd.read_csv(csv_dir + '/train_patients.csv')
#             epoch_df = pd.read_csv(csv_dir + '/train_epochs.csv')
#         elif split == "val":
#             patient_df = pd.read_csv(csv_dir + '/val_patients.csv')
#             epoch_df = pd.read_csv(csv_dir + '/val_epochs.csv')
#         elif split == "test":
#             patient_df = pd.read_csv(csv_dir + '/test_patients.csv')
#             epoch_df = pd.read_csv(csv_dir + '/test_epochs.csv')
        
#         self.epoch_df = epoch_df

#         self.patient_df = patient_df.set_index("nsrrid")

        
#         # ───── LRU cache for whole-night signals ─────
#         self._load_patient_array = lru_cache(maxsize=cache_size)(
#             self._load_patient_array
#         )
#         self.train_edf_cols = train_edf_cols
        
#         self.cache_dir = Path('/u/ztshuai/ondemand/sleep-unimodal/path_cache')
#         self.split = split
#         self._path_cache = self._build_path_cache()

#     # ───────── Dataset API ─────────

#     def __len__(self) -> int:
#         return len(self.epoch_df)

#     def __getitem__(self, idx: int):
        
#         row = self.epoch_df.iloc[idx]
#         nsrrid = row["nsrrid"]
#         epoch_id = int(row["epoch_id"])

#         # 1) load full-night signal and slice to this epoch
#         channel_indicator = pd.read_csv(row['path_head'] + '_channel_indicator.csv')

#         dfs = []
#         for col_name in self.train_edf_cols:

#             if channel_indicator.loc[0, col_name] == 0:
#                 real_index = pd.Index(np.arange(temp_sig.index.min(), np.ceil(temp_sig.index.max()), 1 / TARGET_RATE), name="sec") # incase: 810.0-> 839.9921875
#                 sig_df = pd.DataFrame(
#                     np.zeros(real_index.size, dtype=float),
#                     index=real_index,
#                     columns=[col_name]          
#                 )
#             else:
#                 # overall: 8 batchsize -> 200G - 8 num_of_workers: 0.05~0.1s
#                 # t00 = time.perf_counter() 
#                 temp_sig = self._load_patient_array(nsrrid, row["path_head"], col_name = col_name, epoch_id = epoch_id)

#                 ##############################################################
#                 # 0.003 ~ 0.005 sec
#                 # t01 = time.perf_counter() 
#                 # resample to a shared frequency
#                 TARGET_RATE = self.sample_rate                    # e.g. 128 Hz
#                 period_ms   = 1000 / TARGET_RATE                  # 7.812 ms
#                 samples_per_epoch = self.window_size * TARGET_RATE              # 3840
                
#                 real_index = pd.Index(np.arange(temp_sig.index.min(), np.ceil(temp_sig.index.max()), 1 / TARGET_RATE)) # incase: 810.0-> 839.9921875
                
#                 sig_df = interpolate_index(temp_sig, real_index,
#                                   method="linear", squeeze=False)
#                 # t02 = time.perf_counter() 
#                 # print(f"{col_name} load: {t01 - t00:.3f} s")
#                 # print(f"{col_name} resample: {t02 - t01:.3f} s")
#             ##############################################################
            
            
#             dfs.append(sig_df)
        
#         full_sig = pd.concat(dfs, axis = 1)
        
#         x = torch.tensor(full_sig.values, dtype=torch.float32)
        
#         # 1.5) can add other transformation here
#         if self.transform:
#             x = self.transform(x)
#         # transpose to (C, T)
#         x = x.transpose(0, 1)
#         # 2) add patient-level label(s) if requested
        
#         if self.target_cols:
#             y = torch.tensor(
#                 self.patient_df.loc[nsrrid, self.target_cols].values.astype(float),
#                 dtype=torch.float32,
#             )
#             data = {
#                 'psg': x,
#                 'label': y,
#             }
#             return data
#         else:
#             data = {
#                 'psg': x,
#             }
#             return data
#     def _build_signal_path(self, path_head, col_name='ECG', epoch_id=0, nsrrid=None):
#         """
#         先尝试查缓存；若 key 缺失再退回字符串拼接。
#         """
#         key = (nsrrid, epoch_id, col_name)
#         if key in self._path_cache:
#             return self._path_cache[key]
#         print("cache not hit")
#         # fallback —— 极端情况下 cache 不命中
#         return Path(f"{path_head}/epoch-{epoch_id:05d}_{col_name}.npz")

# #     def _build_signal_path(self, path_head: str, col_name = 'ECG', epoch_id = 0) -> Path:
# #         """
# #         Convert path_head to the actual signal file path.
# #         Default: '<path_head>_data.npz' with key 'signal'.
# #         Adjust if your filenames / keys differ.
# #         """
        
# #         return Path(path_head + f"/epoch-{epoch_id:05d}_{col_name}.npz")
#     def _build_path_cache(self, force_rebuild: bool = False):

#         self.cache_dir.mkdir(parents=True, exist_ok=True)
#         cache_fp = self.cache_dir / f"path_cache_{self.split}.pkl.gz"

        
#         if cache_fp.is_file() and not force_rebuild:
#             with gzip.open(cache_fp, "rb") as fh:
#                 cache = pickle.load(fh)

#             # 用行数 + Row-hash 做快速一致性校验
#             h = xxhash.xxh64(
#                 pd.util.hash_pandas_object(self.epoch_df["path_head"]).values
#             ).intdigest()

#             if cache["meta"]["row_hash"] == h:
#                 return cache["paths"]      

#         paths = {}
#         for idx, row in enumerate(self.epoch_df.itertuples(), 1):
#             for col in self.train_edf_cols:
#                 paths[(row.nsrrid, row.epoch_id, col)] = Path(
#                     f"{row.path_head}/epoch-{row.epoch_id:05d}_{col}.npz"
#                 )
#             if idx % 10_000 == 0:
#                 print(f"gathered {idx:,} hardware paths")

#         meta = {
#             "created_at": time.time(),
#             "row_hash": h,
#             "num_items": len(paths),
#         }
#         with gzip.open(cache_fp, "wb") as fh:
#             pickle.dump({"meta": meta, "paths": paths}, fh,
#                         protocol=pickle.HIGHEST_PROTOCOL)

#         return paths
#     def _load_patient_array(self, nsrrid: str, path_head: str, col_name = 'ECG', epoch_id = 0) -> np.ndarray:
#         """
#         Load the full-night signal into a NumPy array.
#         If you have multiple channels, return shape (C, T).
#         """
        
#         fp = self._build_signal_path(path_head,
#                              col_name=col_name,
#                              epoch_id=epoch_id,
#                              nsrrid=nsrrid)
        
#         if not fp.is_file():
#             raise FileNotFoundError(f"Signal file missing: {fp}")
        
#         with np.load(fp, allow_pickle=True) as npz:
            
#             data = npz['values']
#             index = npz['index']

#             df_stg = pd.DataFrame(
#                 data,
#                 columns=[col_name]
#             )
#             df_stg.insert(0, "sec", index)
            
#             sig = df_stg.set_index("sec")           
        
#         return sig.astype(np.float32)

    
    
    

if __name__ == "__main__":
    # from melp.datasets.augmentations import TRandomResizedCrop, TTimeOut
    # rr_crop_ratio_range = [0.5, 1.0]
    # output_size = 250*5
    # to_crop_ratio_range = [0, 0.5]
    # transforms = [
    #     TRandomResizedCrop(
    #     crop_ratio_range=rr_crop_ratio_range, 
    #     output_size=output_size),
    #     TTimeOut(crop_ratio_range=to_crop_ratio_range)
    # ]
    dataset = ECG_Text_Dataset(split="test", dataset_dir="/data1/r20user2/ECG/raw",  
                               dataset_list=["mimic-iv-ecg"],
                               use_cmsc=False,
                               use_rlm=False,
                               data_pct=0.1,
                               use_ecg_patch=False,
                               num_beats=1
                               )
    print(len(dataset))
    sample = dataset[0]
    # print(sample["ecg_patch"].shape)
    # print(sample["t_indices"].shape)
    ipdb.set_trace()