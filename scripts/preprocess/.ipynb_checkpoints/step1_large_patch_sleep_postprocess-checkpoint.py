import pandas as pd
import numpy as np
import os
from wav2sleep.config import *
# version 1: only master csv
import pandas as pd
from pathlib import Path
from itertools import chain   

SELECT_DATASET = [SHHS]
SOURCE_PROJECT_PATH = 'real_sleep_preprocessed_data'
TARGET_PROJECT_PATH = 'real_sleep_postprocessed_data'
PREPROCESSED_META_PATH = '/projects/besp/shared_data'
dfs = []

####################################################################################
# here we use nsrrid as the unique patient id (we process this in master csv)
# and then we will get epoch id (in the post processing)

for name in SELECT_DATASET:
    dataset = MASTER_CSV_LIST[name]
    for file_path in dataset:
        df_temp = pd.read_csv(file_path)
        df_temp['dataset_name'] = name
        
        storage_path = '/' + SOURCE_PROJECT_PATH + '/' + 'preprocessed_' + name + '/' + name + '/ingest/'# to be decided
        
        
        if name == SHHS:
            def build_path(row):
                if row["visitnumber"] == 1:
                    return PREPROCESSED_META_PATH + storage_path + "shhs1-" + f"{row.nsrrid}"
                if row["visitnumber"] == 2:
                    return PREPROCESSED_META_PATH + storage_path + "shhs2-" + f"{row.nsrrid}"
                return None         
            def build_nsrrid(row):
                if row["visitnumber"] == 1:
                    nsrrid = 'shhs1-' + str(row['nsrrid'])
                if row["visitnumber"] == 2:
                    nsrrid = 'shhs2-' + str(row['nsrrid'])
                return nsrrid
            df_temp["path_head"] = df_temp.apply(build_path, axis=1)
            df_temp['nsrrid'] = df_temp.apply(build_nsrrid, axis=1)          
            df_temp['dataset_id'] = SHHS
        elif name == CHAT:
            # we don't use non-randomized
            def build_path(row):
                if row["vnum"] == 10:
                    return PREPROCESSED_META_PATH + storage_path + "chat-followup-" + f"{row.nsrrid}"
                if row["vnum"] == 3:
                    return PREPROCESSED_META_PATH + storage_path + "chat-baseline-" + f"{row.nsrrid}"
                return None         
            
            def build_nsrrid(row):
                if row["vnum"] == 10:
                    nsrrid = 'chat' + '-followup-' + str(row['nsrrid'])
                if row["vnum"] == 3:
                    nsrrid = 'chat' + '-baseline-' + str(row['nsrrid'])
                return nsrrid
            
            df_temp["path_head"] = df_temp.apply(build_path, axis=1)
            df_temp['nsrrid'] = df_temp.apply(build_nsrrid, axis=1)    
            df_temp['dataset_id'] = CHAT
        else:
            raise ValueError("no matching dataset error code: a08rh293bidbg1")
        
        print(df_temp.shape)
        dfs.append(df_temp)


df = pd.concat(dfs, axis=0)
df.to_csv(PREPROCESSED_META_PATH + '/' + TARGET_PROJECT_PATH + '/example_patient_level_master.csv')
df.set_index("nsrrid", inplace=True)





############## start parallel ##################

# version 2: has event labels, epoch id, ready for training
import pandas as pd
from pathlib import Path
from itertools import chain   
from copy import deepcopy

def load_patient_array(nsrrid: str, path_head: str, col_name = 'ECG') -> np.ndarray:
        """
        Load the full-night signal into a NumPy array.
        If you have multiple channels, return shape (C, T).
        """
        fp = Path(path_head + f"_{col_name}.npz")
        if not fp.is_file():
            raise FileNotFoundError(f"Signal file missing: {fp}")
        with np.load(fp, allow_pickle=False) as npz:
            data = npz['values']
            index = npz['index']

            df_stg = pd.DataFrame(
                data,
                columns=[col_name]
            )
            df_stg.insert(0, "sec", index)
            
            sig = df_stg.set_index("sec")           
            
        return sig.astype(np.float32)
    
####################################################################################
# here we use nsrrid as the unique patient id (we process this in master csv)
# and then we will get epoch id (in the post processing)
if 'nsrrid' in df.columns:
    df.set_index("nsrrid", inplace=True)
CHANNELS = [ECG, HR, PPG,
            SPO2, OX, ABD, THX, AF, NP, SN, 
            EOG_E1_A2, EOG_E2_A1,
            EMG_LLeg, EMG_RLeg, EMG_Chin,
            EEG_C3_A2, EEG_C4_A1, EEG_F3_A2, EEG_F4_A1, EEG_O1_A2, EEG_O2_A1,
            POS,
            # EEG_F4_A1_multitaper_Alpha, EEG_F4_A1_multitaper_Beta, EEG_F4_A1_multitaper_Delta, EEG_F4_A1_multitaper_Gamma, EEG_F4_A1_multitaper_Theta,
           ] # for testing usage
FREQ_CHANNELS = {
    ECG: FREQ_ECG,
    HR: FREQ_HR,
    PPG: FREQ_PPG,
    SPO2: FREQ_SPO2,
    OX: FREQ_OX,
    ABD: FREQ_ABD,
    THX: FREQ_THX,
    AF: FREQ_AF,
    NP: FREQ_NP,
    SN: FREQ_SN,
    EOG_E1_A2: FREQ_EOG_E1_A2,
    EOG_E2_A1: FREQ_EOG_E2_A1,
    EMG_LLeg: FREQ_EMG_LLeg,
    EMG_RLeg: FREQ_EMG_RLeg,
    EMG_Chin: FREQ_EMG_Chin,
    EEG_C3_A2: FREQ_EEG_C3_A2,
    EEG_C4_A1: FREQ_EEG_C4_A1,
    EEG_F3_A2: FREQ_EEG_F3_A2,
    EEG_F4_A1: FREQ_EEG_F4_A1,
    EEG_O1_A2: FREQ_EEG_O1_A2,
    EEG_O2_A1: FREQ_EEG_O2_A1,
    POS: FREQ_POS,
    
}
SAMPLES_PER_EPOCH = 30 # seconds, for testing usage
SAMPLES_PER_SEGMENT = 30 * 5 * 2

def process_patient(idx):
    path_head = df.loc[idx, 'path_head']
    events_path = Path(path_head + '_events.npz')
    stg_path = Path(path_head + '_sleepstage.npz')
    if not events_path.is_file() or not stg_path.is_file():
        return None
    events = np.load(events_path, allow_pickle=True)
    stg = np.load(stg_path, allow_pickle=True)

    data = events['values']
    df_evt = pd.DataFrame(
        data,
        columns=[EVENT_NAME_COLUMN, START_TIME_COLUMN, END_TIME_COLUMN]
    )
    data = stg['values']
    index = stg['index']
    
    df_stg = pd.DataFrame(
        data,
        columns=["Stage"]
    )
    df_stg.insert(0, "sec", index)
    df_stg['epoch_id'] = (df_stg['sec'] // SAMPLES_PER_EPOCH).astype(int)
    df_stg['seg_id'] = (df_stg['sec'] // SAMPLES_PER_SEGMENT).astype(int)
    tib_row = (
        df_evt.loc[df_evt[EVENT_NAME_COLUMN] == 'time_in_bed',
                   [START_TIME_COLUMN, END_TIME_COLUMN]]
        .astype(float)          
        .iloc[0]                
    )
    start_ep = int(tib_row[START_TIME_COLUMN] // SAMPLES_PER_EPOCH)
    end_ep   = int(tib_row[END_TIME_COLUMN]   // SAMPLES_PER_EPOCH)

    df_stg_valid = df_stg.query("@start_ep <= epoch_id <= @end_ep").copy()
    patient_meta = df.loc[[idx]]                
    
    df_epoch = df_stg_valid.merge(patient_meta, how="cross")
    df_epoch['nsrrid'] = idx
    

    EVENT2COL = {
        RESPIRATORY_EVENT_CENTRAL_APNEA:               "Central Apnea",
        RESPIRATORY_EVENT_OBSTRUCTIVE_APNEA:           "Obstructive Apnea",
        RESPIRATORY_EVENT_MIXED_APNEA:                 "Mixed Apnea",
        RESPIRATORY_EVENT_HYPOPNEA:                    "Hypopnea",
        RESPIRATORY_EVENT_DESATURATION:         "Oxygen Desaturation",
        LIMB_MOVEMENT_ISOLATED:      "Limb Movement Isolated",
        LIMB_MOVEMENT_PERIODIC:      "Limb Movement Periodic",
        LIMB_MOVEMENT_ISOLATED_LEFT: "Left Limb Movement Isolated",
        LIMB_MOVEMENT_ISOLATED_RIGHT:"Right Limb Movement Isolated",
        LIMB_MOVEMENT_PERIODIC_LEFT: "Left Limb Movement Periodic",
        LIMB_MOVEMENT_PERIODIC_RIGHT:"Right Limb Movement Periodic",
        AROUSAL_EVENT_CLASSIC:                     "Arousal",
        AROUSAL_EVENT_RESPIRATORY:                        "RERA",
        AROUSAL_EVENT_EMG:         "EMG-Related Arousal",
    }

    IGNORE_EVT = {"time_in_bed", "clock_in_bed"}        

    df_evt_lbl = (
        df_evt.loc[~df_evt[EVENT_NAME_COLUMN].isin(IGNORE_EVT)].copy()
    )
    df_evt_lbl[[START_TIME_COLUMN, END_TIME_COLUMN]] = df_evt_lbl[
        [START_TIME_COLUMN, END_TIME_COLUMN]
    ].astype(float)
    df_evt_lbl["ep_start"] = (df_evt_lbl[START_TIME_COLUMN] // SAMPLES_PER_EPOCH).astype(int) + 1
    df_evt_lbl["ep_end"]   = np.ceil(df_evt_lbl[END_TIME_COLUMN]  / SAMPLES_PER_EPOCH).astype(int)
    df_evt_lbl["epoch_range"] = df_evt_lbl.apply(
        lambda r: range(r.ep_start, r.ep_end + 1), axis=1
    )

    df_evt_long = (
        df_evt_lbl[["EVENT", "epoch_range"]]
        .explode("epoch_range", ignore_index=True)
        .rename(columns={"epoch_range": "epoch_id"})
        .query("EVENT in @EVENT2COL.keys()")
    )

    df_hot = (
        pd.get_dummies(df_evt_long["EVENT"])      
          .groupby(df_evt_long["epoch_id"])
          .max()                                  
          .rename(columns=EVENT2COL)              
          .reindex(columns=EVENT2COL.values())    
          .fillna(0)
          .astype(int)
          .reset_index()
    )

    df_epoch = df_epoch[['nsrrid', 'epoch_id', 'path_head', 'seg_id', 'Stage']]
    df_epoch = df_epoch.merge(df_hot, on="epoch_id", how="left").fillna(0).astype(
        {col: int for col in EVENT2COL.values()}
    )
    
    df_epoch['train_id'] = df_epoch['nsrrid'] + '-epoch-' + df_epoch['epoch_id'].astype(str)
    df_epoch.set_index("train_id", inplace=True)
    
    
    ################################ segment patient data
    path_head = df.loc[idx, "path_head"]
    # path_head = path_head.replace("biosignal_gen_toy_data", "biosignal_gen_real_data")
    # ---------- A) load each channel once ----------
    
    # chan_arrays = {
    #     ch: load_patient_array(idx, path_head, col_name=ch).astype(np.float32)
    #     for ch in CHANNELS                    # e.g. ["ECG", "EEG", "PPG"]
    # }
    chan_arrays = {}
    for ch in CHANNELS:
        fp_temp = Path(path_head + f"_{ch}.npz")
        if not os.path.exists(fp_temp):
            continue
        chan_arrays[ch] = load_patient_array(idx, path_head, col_name=ch).astype(np.float32)
            
    # ---------- B) iterate over epochs ----------
    

    for seg_id in df_epoch["seg_id"]:
        start = (seg_id - 1) * SAMPLES_PER_SEGMENT
        end   = start + SAMPLES_PER_SEGMENT
        is_skip_store = 0
        
        fp_source = path_head + f'/epoch-{seg_id:05d}_all.parquet'
        fp = fp_source.replace(SOURCE_PROJECT_PATH, TARGET_PROJECT_PATH)
        Path(fp).parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(fp): ####################### just for regenerate csv, mask this

            continue
        dfs = {}
        for ch, full_sig in chan_arrays.items():
            
            # print(full_sig.shape)
            ep = full_sig[start:end]
            ep = ep.iloc[:-1]
            
            if ep.shape[0] < SAMPLES_PER_SEGMENT * FREQ_CHANNELS[ch]:      # zero-pad
                is_skip_store = 1
                break
            
            
            dir_path = f"{path_head}" 
            dfs[ch] = pd.DataFrame(ep.values.astype('float32'), index = ep.index, columns=[ch])
            

        if not dfs:
            
            df_epoch.drop(
                df_epoch.index[df_epoch["seg_id"] == seg_id],  # 
                inplace=True
            )
            continue
        if is_skip_store == 1:
            
            df_epoch.drop(
                df_epoch.index[df_epoch["seg_id"] == seg_id],  # 
                inplace=True
            )
            continue
        else:
            df_all_ch_per_epoch = pd.concat(dfs.values(), axis=1)
            
            
            # df_all_ch_per_epoch.to_parquet(fp.replace(SOURCE_PROJECT_PATH, TARGET_PROJECT_PATH)) ####################### just for regenerate csv, mask this
        if not os.path.exists(fp):
            # final check
            df_epoch.drop(
                df_epoch.index[df_epoch["seg_id"] == seg_id],  # 
                inplace=True
            )
            
    print(f"finished patient {idx}")
    ################################ segment patient data
    df_epoch["path_head"] = df_epoch["path_head"].str.replace(SOURCE_PROJECT_PATH, TARGET_PROJECT_PATH)

    return df_epoch
    

print("start processing")
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=8) as ex:
    future_list = [
        ex.submit(process_patient, idx)
        for idx in df.index
    ]

    results = []
    for fut in tqdm(future_list, total=len(future_list)):
        res = fut.result()
        if res is not None:
            results.append(res)

# results = []
# for idx in df.index:
#     res = process_patient(idx)
#     results.append(res)
    
    
    
    
df_epoch_level_all = pd.concat(results, axis=0)

############### end parallel


df_epoch_level_all.to_csv(PREPROCESSED_META_PATH + '/' + TARGET_PROJECT_PATH + '/df_epoch_level_all.csv')