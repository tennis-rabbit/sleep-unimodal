from pathlib import Path
from typing import Dict, Sequence, Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split


def make_patient_epoch_splits(
    patient_df: pd.DataFrame,
    epoch_df:   pd.DataFrame,
    *,
    pretrain_ratio: float = 0.80,          
    downstream_fracs: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    random_state: int = 1337,
    save_dir = None  
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    split patient_df / epoch_df to
        ├─ pretrain
        └─ downstream
              ├─ train
              ├─ val
              └─ test

    """

    if isinstance(patient_df, (str, Path)):
        patient_df = pd.read_csv(patient_df)
    patient_df = patient_df.set_index("nsrrid")


    ids_all          = patient_df.index.unique().tolist()
    pretrain_ids, ds_ids = train_test_split(
        ids_all,
        test_size=1 - pretrain_ratio,
        random_state=random_state,
    )

    ds_train_ids, ds_temp_ids = train_test_split(
        ds_ids,
        test_size=downstream_fracs[1] + downstream_fracs[2],
        random_state=random_state,
    )
    rel_val = downstream_fracs[1] / (downstream_fracs[1] + downstream_fracs[2])
    ds_val_ids, ds_test_ids = train_test_split(
        ds_temp_ids,
        test_size=1.0 - rel_val,
        random_state=random_state,
    )

    split_ids = {
        "pretrain":   pretrain_ids,
        "train":      ds_train_ids,
        "val":        ds_val_ids,
        "test":       ds_test_ids,
    }


    def _sub(df: pd.DataFrame, ids: Sequence[str]) -> pd.DataFrame:
        return df[df["nsrrid"].isin(ids)].reset_index(drop=True)

    results = {
        "pretrain": {
            "patients": patient_df.loc[pretrain_ids].reset_index(),
            "epochs":   _sub(epoch_df, pretrain_ids),
        },
        "downstream": {
            "train": {
                "patients": patient_df.loc[ds_train_ids].reset_index(),
                "epochs":   _sub(epoch_df, ds_train_ids),
            },
            "val": {
                "patients": patient_df.loc[ds_val_ids].reset_index(),
                "epochs":   _sub(epoch_df, ds_val_ids),
            },
            "test": {
                "patients": patient_df.loc[ds_test_ids].reset_index(),
                "epochs":   _sub(epoch_df, ds_test_ids),
            },
        },
    }


    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # helper
        def _save(df: pd.DataFrame, name: str):
            df.to_csv(save_dir / f"{name}.csv", index=False)

        _save(results["pretrain"]["patients"], "pretrain_patients")
        _save(results["pretrain"]["epochs"],   "pretrain_epochs")

        for split in ("train", "val", "test"):
            _save(results["downstream"][split]["patients"], f"{split}_patients")
            _save(results["downstream"][split]["epochs"],   f"{split}_epochs")

    return results


epoch_df = pd.read_csv('/projects/besp/shared_data/real_sleep_postprocessed_data/df_epoch_level_all.csv')
patient_df = pd.read_csv('/projects/besp/shared_data/real_sleep_postprocessed_data/example_patient_level_master.csv')
splits = make_patient_epoch_splits(
    patient_df=patient_df,
    epoch_df=epoch_df,
    pretrain_ratio=0.8,
    save_dir="/projects/besp/shared_data/real_sleep_postprocessed_data/five_min_sleep_data_split_test_with_cache"      # 自动写入 5 个 csv
)
