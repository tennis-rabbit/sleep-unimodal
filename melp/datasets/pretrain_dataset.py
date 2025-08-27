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
#         csv_dir = '/scratch/besp/shared_data/five_min_sleep_data_split_test_with_cache',
#         split: str = "train",
#         *,
#         patient_cols: Optional[Union[str, Sequence[str]]] = None,
#         event_cols: Optional[Union[str, Sequence[str]]] = None,
#         train_edf_cols = None,
#         test_size: float = 0.15,
#         val_size: float = 0.15,
#         random_state: int = 1337,
#         sample_rate: int = 128,
#         window_size: int = 300,
#         epoch_length: int = 30,
#         cache_size: int = 8,
#         transform=None,
#         split_ids: Optional[Dict[str, Sequence[str]]] = None,
#     ):
#         assert split in {"pretrain", "train", "val", "test"}
#         self.transform = transform
#         self.sample_rate = sample_rate
#         self.window_size = window_size
#         self.epoch_length = epoch_length
#         self.patient_cols = (
#             [patient_cols] if isinstance(patient_cols, str) else patient_cols
#         )
#         self.event_cols = (
#             [event_cols] if isinstance(event_cols, str) else event_cols
#         )
#         # ───── patient-level DataFrame ─────
        
#         if split == "pretrain":
#             patient_df = pd.read_csv(csv_dir + '/pretrain_patients.csv')
#             epoch_df = pd.read_csv(csv_dir + '/pretrain_epochs.csv')
#             epoch_df = epoch_df[['nsrrid', 'seg_id', 'epoch_id', 'path_head']]
#         elif split == "train":
#             patient_df = pd.read_csv(csv_dir + '/val_patients.csv')
#             epoch_df = pd.read_csv(csv_dir + '/val_epochs.csv')
#         elif split == "val":
#             patient_df = pd.read_csv(csv_dir + '/val_patients.csv')
#             epoch_df = pd.read_csv(csv_dir + '/val_epochs.csv')
#         elif split == "test":
#             patient_df = pd.read_csv(csv_dir + '/test_patients.csv')
#             epoch_df = pd.read_csv(csv_dir + '/test_epochs.csv')
#         print(epoch_df.columns)
#         self.split = split
#         self.epoch_df = epoch_df
#         if self.split == "pretrain":
#             self.epoch_df = self.epoch_df.drop_duplicates(subset=["seg_id", "nsrrid"], keep="first").reset_index()
#             print(f"{split} length: {len(self.epoch_df)} 5 min segments")
#         else:
            
#             if self.event_cols:
                
#                 if self.event_cols[0] == 'Hypopnea':
#                     self.num_classes = 2
#                 elif self.event_cols[0] == 'Stage':
#                     self.num_classes = 4
#                 else:
#                     self.num_classes = 2
#                 print("33333", self.num_classes)
#             else:
#                 self.num_classes = 2
#             if self.event_cols and ('Stage' in self.event_cols) and ('Stage' in epoch_df.columns):
#                 # 标记无效 epoch
#                 bad_mask = epoch_df['Stage'] == -1
#                 removed_nsrrids = epoch_df.loc[bad_mask, 'nsrrid'].unique().tolist()

#                 # 记录被移除的 nsrrid，便于之后查看
#                 self.removed_nsrrids = removed_nsrrids

#                 # 1) 丢掉 Stage == -1 的行
#                 epoch_df = epoch_df.loc[~bad_mask].copy()

#                 # 2) 为了避免病人级/epoch级数据不一致，进一步丢掉这些病人的所有 epoch（推荐）
#                 if removed_nsrrids:
#                     epoch_df = epoch_df.loc[~epoch_df['nsrrid'].isin(removed_nsrrids)].copy()

#                 # 3) 从 patient_df 中删除这些病人
#                 if removed_nsrrids and ('nsrrid' in patient_df.columns):
#                     patient_df = patient_df.loc[~patient_df['nsrrid'].isin(removed_nsrrids)].copy()

                
#                 self.epoch_df = epoch_df    
        
#         self.patient_df = patient_df.set_index("nsrrid")
        
        
#         # ───── LRU cache for whole-night signals ─────
#         # self._load_patient_array = lru_cache(maxsize=cache_size)(
#         #     self._load_patient_array
#         # )
#         self.train_edf_cols = train_edf_cols

#     # ───────── Dataset API ─────────

#     def __len__(self) -> int:
#         return len(self.epoch_df)
    

#     def _resample_df(self, df: pd.DataFrame, target_hz: int) -> pd.DataFrame:

#         if not np.issubdtype(df.index.dtype, np.number):

#             t = np.arange(len(df)) / float(target_hz)
#             df = df.copy()
#             df.index = t


#         t0 = float(df.index.min())
#         t1 = float(df.index.max())

#         t_target = np.arange(t0, t0 + self.window_size, 1.0 / target_hz)

#         if t_target[-1] > t1:
#             t_target = t_target[t_target <= t1 + 1e-9]


#         out = df.reindex(t_target).interpolate(method="linear", limit_direction="both")

#         out = out.fillna(0.0)
#         return out

#     def __getitem__(self, idx: int):
#         if self.split == "pretrain":
#             row = self.epoch_df.iloc[idx]
#             nsrrid = row["nsrrid"]
#             epoch_id = int(row["seg_id"])


#             cols = list(self.train_edf_cols) if self.train_edf_cols is not None else None
#             df_epoch = self._load_epoch_all_df(row["path_head"], epoch_id, columns=cols)

#             df_epoch = self._resample_df(df_epoch, self.sample_rate)
#             if df_epoch.isna().any().any():
#                 raise ValueError(f"NaN detected in sample idx={idx}, nsrrid={nsrrid}, seg_id={epoch_id}")


#             if cols is not None:
#                 for ch in cols:
#                     if ch not in df_epoch.columns:
#                         # print("missing channel", ch)
#                         df_epoch[ch] = 0.0  
#             df_epoch = df_epoch[cols]  

#             samples_per_epoch = int(self.window_size * self.sample_rate)

#             if len(df_epoch) < samples_per_epoch:
#                 pad = samples_per_epoch - len(df_epoch)
#                 tail = pd.DataFrame({c: 0.0 for c in df_epoch.columns},
#                                     index=df_epoch.index[-1] + (np.arange(1, pad+1) / self.sample_rate))
#                 df_epoch = pd.concat([df_epoch, tail], axis=0)
#             elif len(df_epoch) > samples_per_epoch:
#                 df_epoch = df_epoch.iloc[:samples_per_epoch]
#             if df_epoch.isna().any().any():
#                 raise ValueError(f"NaN detected in sample idx={idx}, nsrrid={nsrrid}, seg_id={epoch_id}")

#             # (T, C) -> tensor -> (C, T)
#             # print(df_epoch.shape)
#             x = torch.tensor(df_epoch.to_numpy(copy=False), dtype=torch.float32).t().contiguous()


#             x = torch.clamp(x, min=-30, max=30)

#             if self.patient_cols:
#                 y = torch.tensor(
#                     self.patient_df.loc[nsrrid, self.patient_cols].values.astype(float),
#                     dtype=torch.float32,
#                 )
#                 return {"psg": x, "label": y.long()}
#             else:
#                 return {"psg": x}
#         else:
#             row = self.epoch_df.iloc[idx]
#             nsrrid = row["nsrrid"]
#             seg_id = int(row["seg_id"])
            
            
#             #############################################
#             seg_df = (
#                 self.epoch_df[
#                     (self.epoch_df['nsrrid'] == nsrrid) &
#                     (self.epoch_df['seg_id']  == seg_id)
#                 ].sort_values('epoch_id')
#             )

            
#             idxs = np.flatnonzero(seg_df['epoch_id'].to_numpy() == row['epoch_id'])
#             if len(idxs) == 0:
#                 raise ValueError("epoch_id not found in this segment")
#             k = int(idxs[0])
#             #############################################
            
#             cols = list(self.train_edf_cols) if self.train_edf_cols is not None else None
#             df_epoch = self._load_epoch_all_df(row["path_head"], seg_id, columns=cols)

#             df_epoch = self._resample_df(df_epoch, self.sample_rate)
           


#             if cols is not None:
#                 for ch in cols:
#                     if ch not in df_epoch.columns:
#                         # print("missing channel", ch)
#                         df_epoch[ch] = 0.0  
#             df_epoch = df_epoch[cols]  

#             samples_per_epoch = int(self.window_size * self.sample_rate)

#             if len(df_epoch) < samples_per_epoch:
#                 pad = samples_per_epoch - len(df_epoch)
#                 tail = pd.DataFrame({c: 0.0 for c in df_epoch.columns},
#                                     index=df_epoch.index[-1] + (np.arange(1, pad+1) / self.sample_rate))
#                 df_epoch = pd.concat([df_epoch, tail], axis=0)
#             elif len(df_epoch) > samples_per_epoch:
#                 df_epoch = df_epoch.iloc[:samples_per_epoch]
            

#             # (T, C) -> tensor -> (C, T)
#             # print(df_epoch.shape)
#             x = torch.tensor(df_epoch.to_numpy(copy=False), dtype=torch.float32).t().contiguous()


#             x = torch.clamp(x, min=-30, max=30)
            
            
            
#             #############################################
#             sample_per_epoch = int(self.epoch_length * self.sample_rate)
            
#             x = x[:, k * sample_per_epoch : (k + 1) * sample_per_epoch]
#             #############################################
            
#             if self.patient_cols:
#                 y = torch.tensor(
#                     self.patient_df.loc[nsrrid, self.patient_cols].values.astype(float),
#                     dtype=torch.float32,
#                 )
                
#                 return {"psg": x, "label": y}
#             elif self.event_cols:
#                 y = torch.tensor(row[self.event_cols].values.astype(float), dtype=torch.float32)
#                 # print(x.shape, y.shape)
                
#                 return {"psg": x, "label": y}
#             else:
#                 return {"psg": x}



#     def _build_epoch_all_path(self, path_head: str, epoch_id: int) -> Path:

#         return Path(f"{path_head}/epoch-{epoch_id:05d}_all.parquet")


#     def _load_epoch_all_df(self, path_head: str, epoch_id: int, columns=None) -> pd.DataFrame:

#         fp = self._build_epoch_all_path(path_head, epoch_id)
#         if not fp.is_file():
#             raise FileNotFoundError(f"Parquet missing: {fp}")

#         df = pd.read_parquet(fp)

#         for c in df.columns:
#             if not np.issubdtype(df[c].dtype, np.floating):
#                 with suppress(Exception):
#                     df[c] = df[c].astype(np.float32)
#         return df




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
        data_pct = 1,
        patient_cols: Optional[Union[str, Sequence[str]]] = None,
        event_cols: Optional[Union[str, Sequence[str]]] = None,
        train_edf_cols = None,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 1337,
        sample_rate: int = 128,
        window_size: int = 300,
        epoch_length: int = 30,
        cache_size: int = 8,
        transform=None,
        split_ids: Optional[Dict[str, Sequence[str]]] = None,
    ):
        assert split in {"pretrain", "train", "val", "test"}
        # ---- basic attrs ----
        self.transform = transform
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.epoch_length = epoch_length
        self.patient_cols = [patient_cols] if isinstance(patient_cols, str) else patient_cols
        self.event_cols   = [event_cols]  if isinstance(event_cols, str)  else event_cols
        self.train_edf_cols = train_edf_cols
        self.split = split
        self.data_pct = float(data_pct)

        # ---- read CSVs ----
        if split == "pretrain":
            patient_df = pd.read_csv(f"{csv_dir}/pretrain_patients.csv")
            epoch_df   = pd.read_csv(f"{csv_dir}/pretrain_epochs.csv")
            keep_cols = [c for c in ['nsrrid', 'seg_id', 'epoch_id', 'path_head', 'Stage'] if c in epoch_df.columns]
            epoch_df = epoch_df[keep_cols].copy()
        elif split == "train":
            # 若确实使用 val_* 作为train，可改回去
            try:
                patient_df = pd.read_csv(f"{csv_dir}/train_patients.csv")
                epoch_df   = pd.read_csv(f"{csv_dir}/train_epochs.csv")
            except FileNotFoundError:
                patient_df = pd.read_csv(f"{csv_dir}/val_patients.csv")
                epoch_df   = pd.read_csv(f"{csv_dir}/val_epochs.csv")
        elif split == "val":
            patient_df = pd.read_csv(f"{csv_dir}/val_patients.csv")
            epoch_df   = pd.read_csv(f"{csv_dir}/val_epochs.csv")
        else:  # test
            patient_df = pd.read_csv(f"{csv_dir}/test_patients.csv")
            epoch_df   = pd.read_csv(f"{csv_dir}/test_epochs.csv")

        print(epoch_df.columns.tolist())

        # ---- num_classes ----
        if self.event_cols:
            if self.event_cols[0] == 'Hypopnea':
                self.num_classes = 2
            elif self.event_cols[0] == 'Stage':
                self.num_classes = 4
            else:
                self.num_classes = 2
        else:
            self.num_classes = 2

        # ---- optional: drop Stage == -1 (no patient deletion) ----
        if self.event_cols and ('Stage' in self.event_cols) and ('Stage' in epoch_df.columns):
            before = len(epoch_df)
            epoch_df = epoch_df.loc[epoch_df['Stage'] != -1].copy()
            if before != len(epoch_df):
                print(f"[{split}] drop Stage==-1: {before - len(epoch_df)} rows")

        # ---- build tables ----
        if split == "pretrain":
            # 预训练保持你的语义：不做“长度=10”强约束
            sort_cols = [c for c in ['nsrrid','seg_id','epoch_id'] if c in epoch_df.columns]
            self.all_epoch_df = epoch_df.sort_values(sort_cols).reset_index(drop=True)
            # 每段一行的索引表
            idx_keep_cols = [c for c in ['nsrrid','seg_id','path_head'] if c in self.all_epoch_df.columns]
            self.epoch_df = (
                self.all_epoch_df[idx_keep_cols]
                .drop_duplicates(['nsrrid','seg_id'], keep='first')
                .reset_index(drop=True)
            )
        else:
            # train/val/test：仅保留“长度 = window_size/epoch_length”的段
            expected_len = self.window_size // self.epoch_length  # 300/30=10
            assert {'nsrrid','seg_id'}.issubset(epoch_df.columns), "epoch_df must have nsrrid & seg_id"
            grp = epoch_df.groupby(['nsrrid', 'seg_id']).size().rename('n').reset_index()
            valid_keys = grp.loc[grp['n'] == expected_len, ['nsrrid', 'seg_id']]
            invalid_ct = int((grp['n'] != expected_len).sum())
            if invalid_ct:
                print(f"[{split}] drop {invalid_ct} segments with n != {expected_len}")

            epoch_df_valid = epoch_df.merge(valid_keys, on=['nsrrid','seg_id'], how='inner')

            sort_cols = [c for c in ['nsrrid','seg_id','epoch_id'] if c in epoch_df_valid.columns]
            self.all_epoch_df = epoch_df_valid.sort_values(sort_cols).reset_index(drop=True)

            idx_keep_cols = [c for c in ['nsrrid','seg_id','path_head'] if c in self.all_epoch_df.columns]
            self.epoch_df = (
                self.all_epoch_df[idx_keep_cols]
                .drop_duplicates(['nsrrid','seg_id'], keep='first')
                .reset_index(drop=True)
            )

        # ---- patient-first sampling (reflect to epoch_df & all_epoch_df & patient_df) ----
        if not (0 < self.data_pct <= 1.0):
            raise ValueError(f"data_pct must be in (0,1], got {self.data_pct}")

        if self.data_pct < 1.0:
            # 仅在“有有效段”的病人里采样
            eligible_patients = pd.Index(self.epoch_df['nsrrid'].unique())
            n_total = len(eligible_patients)
            n_keep  = max(1, int(n_total * self.data_pct))

            # 可重复的随机采样
            sampled_nsrrids = (
                pd.Series(eligible_patients)
                .sample(n=n_keep, random_state=random_state, replace=False)
                .to_list()
            )

            # 过滤三张表
            self.epoch_df = self.epoch_df.loc[self.epoch_df['nsrrid'].isin(sampled_nsrrids)].reset_index(drop=True)
            self.all_epoch_df = self.all_epoch_df.loc[self.all_epoch_df['nsrrid'].isin(sampled_nsrrids)].reset_index(drop=True)
            patient_df = patient_df.loc[patient_df['nsrrid'].isin(sampled_nsrrids)].copy()

            print(f"[{split}] patient-sampled {n_keep}/{n_total} (data_pct={self.data_pct})")

        # ---- finalize patient_df index ----
        self.patient_df = patient_df.set_index("nsrrid")

        # ---- summary ----
        if split == "pretrain":
            print(f"[{split}] segments (index rows): {len(self.epoch_df)}; total rows: {len(self.all_epoch_df)}")
        else:
            print(f"[{split}] segments kept: {len(self.epoch_df)}; "
                  f"total epochs: {len(self.all_epoch_df)}; expected per seg: {self.window_size // self.epoch_length}")

#     def __init__(
#         self,
#         csv_dir = '/scratch/besp/shared_data/five_min_sleep_data_split_test_with_cache',
#         split: str = "train",
#         *,
#         data_pct = 1,
#         patient_cols: Optional[Union[str, Sequence[str]]] = None,
#         event_cols: Optional[Union[str, Sequence[str]]] = None,
#         train_edf_cols = None,
#         test_size: float = 0.15,
#         val_size: float = 0.15,
#         random_state: int = 1337,
#         sample_rate: int = 128,
#         window_size: int = 300,
#         epoch_length: int = 30,
#         cache_size: int = 8,
#         transform=None,
#         split_ids: Optional[Dict[str, Sequence[str]]] = None,
#     ):
#         assert split in {"pretrain", "train", "val", "test"}
#         self.transform = transform
#         self.sample_rate = sample_rate
#         self.window_size = window_size
#         self.epoch_length = epoch_length
#         self.patient_cols = [patient_cols] if isinstance(patient_cols, str) else patient_cols
#         self.event_cols   = [event_cols]  if isinstance(event_cols, str)  else event_cols
#         self.train_edf_cols = train_edf_cols
#         self.split = split

#         # ───── read CSVs (按 split) ─────
#         if split == "pretrain":
#             patient_df = pd.read_csv(f"{csv_dir}/pretrain_patients.csv")
#             epoch_df   = pd.read_csv(f"{csv_dir}/pretrain_epochs.csv")
#             keep_cols = [c for c in ['nsrrid', 'seg_id', 'epoch_id', 'path_head'] if c in epoch_df.columns]
#             epoch_df = epoch_df[keep_cols].copy()
#         elif split == "train":
#             # 如果你的文件确实是 val_* 用于训练，可改回去
#             try:
#                 patient_df = pd.read_csv(f"{csv_dir}/train_patients.csv")
#                 epoch_df   = pd.read_csv(f"{csv_dir}/train_epochs.csv")
#             except FileNotFoundError:
#                 patient_df = pd.read_csv(f"{csv_dir}/val_patients.csv")
#                 epoch_df   = pd.read_csv(f"{csv_dir}/val_epochs.csv")
#         elif split == "val":
#             patient_df = pd.read_csv(f"{csv_dir}/val_patients.csv")
#             epoch_df   = pd.read_csv(f"{csv_dir}/val_epochs.csv")
#         else:  # "test"
#             patient_df = pd.read_csv(f"{csv_dir}/test_patients.csv")
#             epoch_df   = pd.read_csv(f"{csv_dir}/test_epochs.csv")

#         print(epoch_df.columns.tolist())

#         # ───── num_classes ─────
#         if self.event_cols:
#             if self.event_cols[0] == 'Hypopnea':
#                 self.num_classes = 2
#             elif self.event_cols[0] == 'Stage':
#                 self.num_classes = 4
#             else:
#                 self.num_classes = 2
#         else:
#             self.num_classes = 2

#         # ───── 可选：清洗掉 Stage == -1 的样本（不改 patient_df） ─────
#         if self.event_cols and ('Stage' in self.event_cols) and ('Stage' in epoch_df.columns):
#             before = len(epoch_df)
#             epoch_df = epoch_df.loc[epoch_df['Stage'] != -1].copy()
#             if before != len(epoch_df):
#                 print(f"[{split}] drop Stage==-1: {before - len(epoch_df)} rows")

#         # ───── 只保留长度满足预期的 (nsrrid, seg_id) 段 ─────
#         expected_len = self.window_size // self.epoch_length  # 300/30=10
#         assert {'nsrrid','seg_id'}.issubset(epoch_df.columns), "epoch_df must have nsrrid & seg_id"
#         grp = epoch_df.groupby(['nsrrid', 'seg_id']).size().rename('n').reset_index()
#         valid_keys = grp.loc[grp['n'] == expected_len, ['nsrrid', 'seg_id']]
#         invalid_ct = int((grp['n'] != expected_len).sum())
#         if invalid_ct:
#             print(f"[{split}] drop {invalid_ct} segments with n != {expected_len}")

#         epoch_df_valid = epoch_df.merge(valid_keys, on=['nsrrid','seg_id'], how='inner')

#         # 全量表：每段的所有 epoch（供 __getitem__ 取 10 条）
#         sort_cols = [c for c in ['nsrrid','seg_id','epoch_id'] if c in epoch_df_valid.columns]
#         self.all_epoch_df = epoch_df_valid.sort_values(sort_cols).reset_index(drop=True)

#         # 段级索引：每段 1 行（决定 __len__/采样）；保留常用列
#         idx_keep_cols = [c for c in ['nsrrid','seg_id','path_head'] if c in self.all_epoch_df.columns]
#         self.epoch_df = (
#             self.all_epoch_df[idx_keep_cols]
#             .drop_duplicates(['nsrrid','seg_id'], keep='first')
#             .reset_index(drop=True)
#         )

#         # 不对 patient_df 做删改
#         self.patient_df = patient_df.set_index("nsrrid")

#         print(f"[{split}] segments kept: {len(self.epoch_df)}; "
#               f"total epochs: {len(self.all_epoch_df)}; expected per seg: {expected_len}")


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
                return {"psg": x, "label": y.long()}
            else:
                return {"psg": x}
        else:
            row = self.epoch_df.iloc[idx]
            nsrrid = row["nsrrid"]
            seg_id = int(row["seg_id"])
            
            
            #############################################
            seg_df = (
                self.all_epoch_df[
                    (self.all_epoch_df['nsrrid'] == nsrrid) &
                    (self.all_epoch_df['seg_id']  == seg_id)
                ].sort_values('epoch_id')
            )
            patient_y_duplicate_num = len(seg_df)
            
            
            # idxs = np.flatnonzero(seg_df['epoch_id'].to_numpy() == row['epoch_id'])
            # if len(idxs) == 0:
            #     raise ValueError("epoch_id not found in this segment")
            # k = int(idxs[0])
            #############################################
            
            cols = list(self.train_edf_cols) if self.train_edf_cols is not None else None
            df_epoch = self._load_epoch_all_df(row["path_head"], seg_id, columns=cols)

            df_epoch = self._resample_df(df_epoch, self.sample_rate)
           


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
            

            # (T, C) -> tensor -> (C, T)
            # print(df_epoch.shape)
            x = torch.tensor(df_epoch.to_numpy(copy=False), dtype=torch.float32).t().contiguous()


            x = torch.clamp(x, min=-30, max=30)
            
            
            
            #############################################
#             sample_per_epoch = int(self.epoch_length * self.sample_rate)
            
#             x = x[:, k * sample_per_epoch : (k + 1) * sample_per_epoch]
            #############################################
            
            if self.patient_cols:
                y = torch.tensor(
                    self.patient_df.loc[nsrrid, self.patient_cols].values.astype(float),
                    dtype=torch.float32,
                )
                y = y.repeat(10)
                
                return {"psg": x, "label": y}
            elif self.event_cols:
                y = torch.tensor(seg_df[self.event_cols].values.astype(float), dtype=torch.float32).squeeze(1)
                
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