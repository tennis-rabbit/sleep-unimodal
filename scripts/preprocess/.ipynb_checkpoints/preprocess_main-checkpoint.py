######################### Note
############## modify: 1) data path; 2) logics of file structure; 3) parser of .xml or .txt files. 
"""Process the NSRR datasets.

Turns EDF files into parquet files containing sleep labels and the signals used.
"""

import argparse
import logging
import os
from glob import glob
import numpy as np
import pandas as pd
import mne
import ray
from tqdm import tqdm
import shutil

from wav2sleep.data.edf import load_edf_data
from wav2sleep.data.txt_parse_all import parse_txt_annotations_wsc
from wav2sleep.data.txt_parse_all import parse_all_score_wsc, parse_sco_wsc
from wav2sleep.data.utils import interpolate_index, mne_lowpass_series
from wav2sleep.data.xml import parse_xml_annotations
from wav2sleep.settings import *
from wav2sleep.config import *
from wav2sleep.parallel import parallelise
from wav2sleep.data.hsp_dataset import hsp_checkio, prepare_hsp_dataset, download_hsp_subfolder, process_hsp_main

from wav2sleep.data.xml_parse_all import parse_all_annotations
import xml.etree.ElementTree as ET
import re
import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# --- TODO ZITAO
# test npz storage 
# Use the annotation name dictionary in config.py to standardize the annotation names in all_df of each dataset
    # need to add all alternative annotation names for each dataset
# Add other channels to the channel name conversion dictionary in config.py, make sure this is comprehensive for all datasets
# ---
def normalize_event_concepts(df: pd.DataFrame,
                             mapping: dict = EVENT_NAME_MAPPING,
                             *,
                             column: str = "EVENT",
                             case_insensitive: bool = True,
                             strip_space: bool = True,
                             unmapped_action: str = "keep",   # "keep", "nan", or a prefix like "UNMAPPED__"
                             inplace: bool = False) -> pd.DataFrame:
    """
    Replace values in `column` with canonical event keys defined in EVENT_NAME_MAPPING.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing an event concept column.
    mapping : dict
        EVENT_NAME_MAPPING dictionary.
    column : str
        Column name to normalize (default "Concepts").
    case_insensitive : bool
        Ignore case when matching aliases.
    strip_space : bool
        Strip leading/trailing spaces before matching.
    unmapped_action : str
        - "keep" : leave unmatched values unchanged (default)
        - "nan"  : set unmatched values to NaN
        - any other string (e.g. "UNMAPPED__") : prefix the original value with this string
    inplace : bool
        If True, modify df in place; else return a new dataframe.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized `column`.
    """
    if column not in df.columns:
        raise KeyError(f"`{column}` not found in dataframe")

    # Build reverse lookup: alias → canonical_key
    reverse_lookup = {}
    for group in mapping.values():
        for canonical_key, aliases in group.items():
            for alias in aliases:
                
                key = str(alias).strip() if strip_space else str(alias)
                if case_insensitive:
                    key = key.lower()
                reverse_lookup[key] = canonical_key

    # Prepare the mapper function
    def map_alias(val: str):
        if pd.isna(val):
            return val  # keep NaN as-is
        key = val.strip() if strip_space else val
        if case_insensitive:
            key = key.lower()
        if key in reverse_lookup:
            return reverse_lookup[key]
        # Unmapped handling
        if unmapped_action == "keep":
            return val
        elif unmapped_action == "nan":
            return np.nan
        else:  # prefix
            return f"{unmapped_action}{val}"

    # Apply mapping
    target = df if inplace else df.copy()
    target[column] = target[column].apply(map_alias)
    return target
# --- TODO both, maybe?
# check PPG or Plethysmography channels in all datasets
# ---

# --- TODO KAI
# investigate annotations logic for all other sites in HSP
    # we can use the file paths in jupyter to explore each of the annotation files and how they are formatted. 
    # try to reuse functions for S0001 as much as possible. 
# Change EEG spec class to make sure output is the same size as input if the step is 1 , take a deeper dive into EEG spec generation, specifically look at Yuzhes code 
# ---


def process_edf(edf: pd.DataFrame):
    """Process dataframe of EDF data."""

    signals = []
    channel_indicator = {COL: 0 for COL in EDF_COLS} # init column indicator to 0 for all columns
        # 1 = present, 0 = missing, 0.5 = bipolar lead manual creation
    
    def _process_edf_column(col, target_index, preprocessed_fs):
        """Process signal column of EDF"""
        if col in edf:
            
            raw = edf[col].dropna()
            
            # print(len(raw.loc[0:1]))
            raw_fs = len(raw.loc[0:1]) - 1 # this gets the sampling rate of the signal, assuming index is in seconds
            
            if raw_fs > preprocessed_fs: 
                if col in [EMG_LLeg, EMG_RLeg, EMG_LLeg1, EMG_LLeg2, EMG_RLeg1, EMG_RLeg2, EMG_Chin, EMG_RChin, EMG_LChin, EMG_CChin]:
                    raw_hp = mne_lowpass_series(raw, raw_fs, cutoff = preprocessed_fs/2, highpass_cutoff = 10)
                else:
                    raw_hp = mne_lowpass_series(raw, raw_fs, cutoff = preprocessed_fs/2)

            else:
                if col in [EMG_LLeg, EMG_RLeg, EMG_LLeg1, EMG_LLeg2, EMG_RLeg1, EMG_RLeg2, EMG_Chin, EMG_RChin, EMG_LChin, EMG_CChin]:
                    raw_hp = mne_lowpass_series(raw, raw_fs, cutoff = None, highpass_cutoff = 10)
                else:
                    raw_hp = raw
            
            real_index = pd.Index(np.arange(0, int(len(edf[col].dropna())/raw_fs) + 1e-9, 1 / preprocessed_fs)[1:])
            
            if len(real_index)<len(target_index):
                
                resampled = interpolate_index(raw_hp, real_index,
                              method="linear", squeeze=False)
            else:
                
                resampled = interpolate_index(raw_hp, target_index,
                              method="linear", squeeze=False)

            signals.append(resampled)
            channel_indicator[col] = 1
            return 1 # WARNING: I changed this to 1 instead of 0
        else:
            channel_indicator[col] = 0
            return 0 # still need to return for bipolar lead logic

    is_available_ECG = _process_edf_column(ECG, ECG_SIGNAL_INDEX, FREQ_ECG)
    is_available_ECG1 = _process_edf_column(ECG1, ECG1_SIGNAL_INDEX, FREQ_ECG1)
    is_available_ECG2 = _process_edf_column(ECG2, ECG2_SIGNAL_INDEX, FREQ_ECG2)
    is_available_ECG3 = _process_edf_column(ECG3, ECG3_SIGNAL_INDEX, FREQ_ECG3)
    
    is_available_HR = _process_edf_column(HR, HR_SIGNAL_INDEX, FREQ_HR)
    is_available_PPG = _process_edf_column(PPG, PPG_SIGNAL_INDEX, FREQ_PPG)

    is_available_SPO2 = _process_edf_column(SPO2, SPO2_SIGNAL_INDEX, FREQ_SPO2)
    is_available_OX = _process_edf_column(OX, OX_SIGNAL_INDEX, FREQ_OX)
    is_available_ABD = _process_edf_column(ABD, ABD_SIGNAL_INDEX, FREQ_ABD)
    is_available_THX = _process_edf_column(THX, THX_SIGNAL_INDEX, FREQ_THX)
    is_available_AF = _process_edf_column(AF, AF_SIGNAL_INDEX, FREQ_AF)
    is_available_NP = _process_edf_column(NP, NP_SIGNAL_INDEX, FREQ_NP)
    is_available_SN = _process_edf_column(SN, SN_SIGNAL_INDEX, FREQ_SN)
    
    is_available_EMG_LLeg = _process_edf_column(EMG_LLeg, EMG_LLeg_SIGNAL_INDEX, FREQ_EMG_LLeg)
    is_available_EMG_RLeg = _process_edf_column(EMG_RLeg, EMG_RLeg_SIGNAL_INDEX, FREQ_EMG_RLeg)
    is_available_EMG_LLeg1 = _process_edf_column(EMG_LLeg1, EMG_LLeg1_SIGNAL_INDEX, FREQ_EMG_LLeg1)
    is_available_EMG_LLeg2 = _process_edf_column(EMG_LLeg2, EMG_LLeg2_SIGNAL_INDEX, FREQ_EMG_LLeg2)
    is_available_EMG_RLeg1 = _process_edf_column(EMG_RLeg1, EMG_RLeg1_SIGNAL_INDEX, FREQ_EMG_RLeg1)
    is_available_EMG_RLeg2 = _process_edf_column(EMG_RLeg2, EMG_RLeg2_SIGNAL_INDEX, FREQ_EMG_RLeg2)

    is_available_EMG_Chin = _process_edf_column(EMG_Chin, EMG_Chin_SIGNAL_INDEX, FREQ_EMG_Chin)
    is_available_EMG_LChin = _process_edf_column(EMG_LChin, EMG_LChin_SIGNAL_INDEX, FREQ_EMG_LChin)
    is_available_EMG_RChin = _process_edf_column(EMG_RChin, EMG_RChin_SIGNAL_INDEX, FREQ_EMG_RChin)
    is_available_EMG_CChin = _process_edf_column(EMG_CChin, EMG_CChin_SIGNAL_INDEX, FREQ_EMG_CChin)

    is_available_E1 = _process_edf_column(EOG_L, EOG_L_SIGNAL_INDEX, FREQ_EOG_L)
    is_available_E2 = _process_edf_column(EOG_R, EOG_R_SIGNAL_INDEX, FREQ_EOG_R)
    is_available_E1_A2 = _process_edf_column(EOG_E1_A2, EOG_E1_A2_SIGNAL_INDEX, FREQ_EOG_E1_A2)
    is_available_E2_A1 = _process_edf_column(EOG_E2_A1, EOG_E2_A1_SIGNAL_INDEX, FREQ_EOG_E2_A1)
    
    
    is_available_C3 = _process_edf_column(EEG_C3, EEG_C3_SIGNAL_INDEX, FREQ_EEG_C3)
    is_available_C4 = _process_edf_column(EEG_C4, EEG_C4_SIGNAL_INDEX, FREQ_EEG_C4)
    is_available_A1 = _process_edf_column(EEG_A1, EEG_A1_SIGNAL_INDEX, FREQ_EEG_A1)
    is_available_A2 = _process_edf_column(EEG_A2, EEG_A2_SIGNAL_INDEX, FREQ_EEG_A2)
    is_available_O1 = _process_edf_column(EEG_O1, EEG_O1_SIGNAL_INDEX, FREQ_EEG_O1)
    is_available_O2 = _process_edf_column(EEG_O2, EEG_O2_SIGNAL_INDEX, FREQ_EEG_O2)
    is_available_F3 = _process_edf_column(EEG_F3, EEG_F3_SIGNAL_INDEX, FREQ_EEG_F3)
    is_available_F4 = _process_edf_column(EEG_F4, EEG_F4_SIGNAL_INDEX, FREQ_EEG_F4)
    
    
    is_available_C3_A2 = _process_edf_column(EEG_C3_A2, EEG_C3_A2_SIGNAL_INDEX, FREQ_EEG_C3_A2)
    is_available_C4_A1 = _process_edf_column(EEG_C4_A1, EEG_C4_A1_SIGNAL_INDEX, FREQ_EEG_C4_A1)
    is_available_F3_A2 = _process_edf_column(EEG_F3_A2, EEG_F3_A2_SIGNAL_INDEX, FREQ_EEG_F3_A2)
    is_available_F4_A1 = _process_edf_column(EEG_F4_A1, EEG_F4_A1_SIGNAL_INDEX, FREQ_EEG_F4_A1)
    is_available_O1_A2 = _process_edf_column(EEG_O1_A2, EEG_O1_A2_SIGNAL_INDEX, FREQ_EEG_O1_A2)
    is_available_O2_A1 = _process_edf_column(EEG_O2_A1, EEG_O2_A1_SIGNAL_INDEX, FREQ_EEG_O2_A1)

    is_available_POS = _process_edf_column(POS, POS_SIGNAL_INDEX, FREQ_POS)
    
    merged_df = pd.concat(signals, axis=1).astype(np.float32)

    # check if bipolar ECG lead needs to be calculated
    if (ECG not in merged_df.columns.to_list()):
        if (is_available_ECG3 == 1) and (is_available_ECG1 == 1):
            merged_df[ECG] = merged_df[ECG3] - merged_df[ECG1]
            channel_indicator[ECG] = 0.5 # indicate bipolar lead manual creation with 0.5
        elif (is_available_ECG2 == 1) and (is_available_ECG1 == 1):
            merged_df[ECG] = merged_df[ECG2] - merged_df[ECG1]
            channel_indicator[ECG] = 0.5 # indicate bipolar lead manual creation with 0.5

    # check if bipolar EMG lead needs to be calculated
    if (EMG_LLeg not in merged_df.columns.to_list()) and (is_available_EMG_LLeg1 == 1) and (is_available_EMG_LLeg2 == 1):
        merged_df[EMG_LLeg] = merged_df[EMG_LLeg1] - merged_df[EMG_LLeg2]
        channel_indicator[EMG_LLeg] = 0.5 # indicate bipolar lead manual creation with 0.5
    if (EMG_RLeg not in merged_df.columns.to_list()) and (is_available_EMG_RLeg1 == 1) and (is_available_EMG_RLeg2 == 1):
        merged_df[EMG_RLeg] = merged_df[EMG_RLeg1] - merged_df[EMG_RLeg2]
        channel_indicator[EMG_RLeg] = 0.5 # indicate bipolar lead manual creation with 0.5

    # check if bipolar EMG chin needs to be calculated
    if (EMG_Chin not in merged_df.columns.to_list()):
        if (is_available_EMG_LChin == 1) and (is_available_EMG_CChin == 1):
            merged_df[EMG_Chin] = merged_df[EMG_LChin] - merged_df[EMG_CChin]
            channel_indicator[EMG_Chin] = 0.5 # indicate bipolar lead manual creation with 0.5
        elif (is_available_EMG_RChin == 1) and (is_available_EMG_CChin == 1):
            merged_df[EMG_Chin] = merged_df[EMG_RChin] - merged_df[EMG_CChin]
            channel_indicator[EMG_Chin] = 0.5 # indicate bipolar lead manual creation with 0.5

    # check if bipolar EEG lead needs to be calculated
    if (EEG_C3_A2 not in merged_df.columns.to_list()) and (is_available_C3 == 1) and (is_available_A2 == 1):
        merged_df[EEG_C3_A2] = merged_df[EEG_C3] - merged_df[EEG_A2]
        channel_indicator[EEG_C3_A2] = 0.5 # indicate bipolar lead manual creation with 0.5
    if (EEG_C4_A1 not in merged_df.columns.to_list()) and (is_available_C4 == 1) and (is_available_A1 == 1):
        merged_df[EEG_C4_A1] = merged_df[EEG_C4] - merged_df[EEG_A1]
        channel_indicator[EEG_C4_A1] = 0.5 # indicate bipolar lead manual creation with 0.5
    if (EEG_F3_A2 not in merged_df.columns.to_list()) and (is_available_F3 == 1) and (is_available_A2 == 1):
        merged_df[EEG_F3_A2] = merged_df[EEG_F3] - merged_df[EEG_A2]
        channel_indicator[EEG_F3_A2] = 0.5 # indicate bipolar lead manual creation with 0.5
    if (EEG_F4_A1 not in merged_df.columns.to_list()) and (is_available_F4 == 1) and (is_available_A1 == 1):
        merged_df[EEG_F4_A1] = merged_df[EEG_F4] - merged_df[EEG_A1]
        channel_indicator[EEG_F4_A1] = 0.5 # indicate bipolar lead manual creation with 0.5
    if (EEG_O1_A2 not in merged_df.columns.to_list()) and (is_available_O1 == 1) and (is_available_A2 == 1):
        merged_df[EEG_O1_A2] = merged_df[EEG_O1] - merged_df[EEG_A2]
        channel_indicator[EEG_O1_A2] = 0.5 # indicate bipolar lead manual creation with 0.5
    if (EEG_O2_A1 not in merged_df.columns.to_list()) and (is_available_O2 == 1) and (is_available_A1 == 1):
        merged_df[EEG_O2_A1] = merged_df[EEG_O2] - merged_df[EEG_A1]
        channel_indicator[EEG_O2_A1] = 0.5 # indicate bipolar lead manual creation with 0.5
        
    if (is_available_E1 == 1) and (is_available_A2 == 1):
        merged_df[EOG_E1_A2] = merged_df[EOG_L] - merged_df[EEG_A2] 
        channel_indicator[EOG_E1_A2] = 0.5
    if (is_available_E2 == 1) and (is_available_A1 == 1):
        merged_df[EOG_E2_A1] = merged_df[EOG_R] - merged_df[EEG_A1]  
        channel_indicator[EOG_E2_A1] = 0.5

    # ######################################
    # # wait for Kai's part
    # if EEG_C3_A2 in merged_df.columns.to_list():
    #     eeg_signal = merged_df[EEG_C3_A2]
    #     eeg_spec = EEGSpec(eeg_signal.values, fs = 64, 
    #                output_mode = 'density', eeg_bands = True, 
    #                detrend = True, powerline_freq = 60)
    #     freq_welch, times_welch, Sxx_welch, band_welch = eeg_spec.welch(frame_win_sec = 20, frame_step_sec = 5, welch_win_sec=4, welch_step_sec=2)
    

    merged_df = (merged_df - merged_df.mean()) / merged_df.std() # should we change to local normalization?, I added to wav2sleep.data.utils.py
    return merged_df, channel_indicator

####################### need to modify here based on the logics in the parser ###########################

def process(edf_fp: str, label_fp: str, output_fp: str, dataset: str, overwrite: bool = False) -> bool:
    """Process night of data."""
    
    # check if file exists
    # output_path_events = output_fp + '_events.parquet'
    # if os.path.exists((output_path_events):
    #     logger.debug(f'Skipping {edf_fp=}, {output_fp=}, already exists')
            # return False
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",output_fp)
    output_fp = np.sqrt(output_fp)
    # --- check overwrite and make directories ---

    if dataset == HSP:
        output_dir = os.path.dirname(output_fp)
        parquet_files = glob(output_fp + '*.parquet') # parquet files in directory for this session
        if os.path.isdir(output_dir) and len(parquet_files) >= 4 and not overwrite:
            # for HSP, we expect _sleepstage.parquet, _POS.parquet, _events.parquet, and raw channel data
            logger.debug(f'Skipping {edf_fp=}, parquet files in {output_dir=} already exists')
            return False
        else:
            os.makedirs(output_dir, exist_ok=True)
    else:
        if os.path.exists(output_fp) and not overwrite:
            logger.debug(f'Skipping {edf_fp=}, {output_fp=}, already exists')
            return False
        else:
            os.makedirs(os.path.dirname(output_fp), exist_ok=True)
    # ---

    if dataset == HSP: # completely different logic for HSP dataset: we need to download the files from aws server and process the annotations very differently 
        # for HSP: edf_fp = file path to store downloaded files, label_fp = patient-level subfolder suffix to download from aws, output_fp = filepath to store output files

        # --- make directory for download ---
        download_folder = edf_fp
        aws_folder = label_fp
        if not os.path.isdir(download_folder):
            os.makedirs(download_folder, exist_ok=True)
        # ---

        # --- download files from aws server ---
        status = download_hsp_subfolder(aws_folder, download_folder)
        if not status:
            logger.error(f'Failed to download {aws_folder} from AWS server.')
            return False
        # --- 

        # --- infer the site number, ---
        site = label_fp.split('/')[0]
        # ---

        # --- invoke site-specific processing logic ---
        edf_fp, labels, body_position_series, all_df, presleep_questionnaire, annotation_indicator_df, error = process_hsp_main(download_folder, site)
        if error:
            logger.error(error)
            return False
        # NOTE: labels is None if sleep stages are missing
        # NOTE: body_position_series is None if body positions are missing
        # NOTE: all_df is None if no other annotation events are present
        # NOTE: these are indicated in annotation_indicator_dict
        annotation_indicator_df.to_csv(output_fp + '_annotation_indicator.csv', index = False)
        # ---

        # --- specific to HSP site S0001, save presleep questionnaire ---
        if site == S0001:
            if presleep_questionnaire is not None: # presleep questionnaire indicator is included in the annotation_indicator_df
                presleep_questionnaire.to_csv(output_fp + '_presleep_questionnaire.csv', index = False)
        # ---

        # --- save body position if it exists ---
        if body_position_series is not None:
            body_position_series = body_position_series.reindex(POS_SIGNAL_INDEX).fillna(BODY_POSITION_OTHER_UNKNOWN) # need this if we are standardizing length to MAX_LENGTH
            body_position_series_df = body_position_series.to_frame()
            body_position_series_df.dropna(inplace = True)
            body_position_series_df.to_parquet(output_fp + f'_{POS}.parquet')   
        # ---

    else: # other datasets are processed here, with similar logic
            
        # --- Process xml sleep stages and annotations ---
        if label_fp.endswith('.xml'):
            try:
                labels = parse_xml_annotations(label_fp) # parse sleep stages
                
                all_df = parse_all_annotations(label_fp) # parse all other events (arousals, respiratory)
                
                def _extract_start_duration_clock(xml_fp,
                                                  target_concept: str = "Recording Start Time",
                                                  parse_clock: bool = True):
                    tree = ET.parse(xml_fp)
                    root = tree.getroot()

                    for evt in root.findall(".//ScoredEvent"):
                        concept = (evt.findtext("EventConcept") or "").strip()
                        if concept != target_concept:
                            continue

                        start    = float(evt.findtext("Start"))
                        duration = float(evt.findtext("Duration"))
                        clock_raw = (evt.findtext("ClockTime") or "").strip() or None

                        if not parse_clock:
                            return start, duration, clock_raw

                        clock_clean = None
                        if clock_raw:

                            m = re.search(r"(\d{2})[.:](\d{2})[.:](\d{2})$", clock_raw)
                            if m:
                                h, m_, s = map(int, m.groups())
                                clock_clean = datetime.time(hour=h, minute=m_, second=s).strftime("%H:%M:%S")

                        return start, duration, clock_clean

                    raise ValueError(f"EventConcept '{target_concept}' not found in {xml_fp}")
                        
                if dataset == SHHS:
                    def extract_id_shhs(path: str) -> str:
                        stem = Path(path).stem                     # 'shhs1-200728-nsrr'
                        return stem.split('-')[1], stem.split('-')[0]                  # '200728'

                    start_time, duration, clock_obj = _extract_start_duration_clock(label_fp)
                    subject_id, visit_id = extract_id_shhs(label_fp)
                    
                    
                    if visit_id == 'shhs1':
                        df = pd.read_csv(RAW_SHHS[0], encoding="cp1252")
                        start_sleep_epoch = df.loc[df['nsrrid'] == int(subject_id), 'stloutp'].iloc[0] # epoch
                        sleep_time_min = df.loc[df['nsrrid'] == int(subject_id), 'timebedp'].iloc[0] # sleep mins
                        start_sleep_time = start_time + 30 * start_sleep_epoch
                        end_sleep_time = start_sleep_time + 60 * sleep_time_min
                    elif visit_id == 'shhs2':
                        df = pd.read_csv(RAW_SHHS[1], encoding="cp1252")
                        start_sleep_clock =df.loc[df['nsrrid'] == int(subject_id), 'stloutp'].iloc[0] # clock time
                        sleep_time_min = df.loc[df['nsrrid'] == int(subject_id), 'timebedp'].iloc[0] # sleep mins
                        # print("sleep time: ", start_sleep_clock, "start collect time: ", clock_obj)
                        start_sleep_time = (
                            datetime.datetime.strptime(start_sleep_clock, "%H:%M:%S")
                            - datetime.datetime.strptime(clock_obj, "%H:%M:%S")
                        ).total_seconds()
                        if start_sleep_time <0:
                            start_sleep_time = start_sleep_time + 86400
                        end_sleep_time = start_sleep_time + 60 * sleep_time_min
                    new_row = {
                            #"Types":    "time_in_bed",
                            EVENT_NAME_COLUMN: "time_in_bed",
                            START_TIME_COLUMN:   start_sleep_time,
                            END_TIME_COLUMN:     end_sleep_time,
                            #"Signal":   "time_in_bed"
                    }
                    all_df.loc[len(all_df)] = new_row 
                    start_sleep_clock = str(datetime.timedelta(seconds=start_sleep_time))[-8:]
                    end_sleep_clock = str(datetime.timedelta(seconds=end_sleep_time))[-8:]
                    new_clock_row = {
                            #"Types":    "time_in_bed",
                            EVENT_NAME_COLUMN: "clock_in_bed",
                            START_TIME_COLUMN:   start_sleep_clock,
                            END_TIME_COLUMN:     end_sleep_clock,
                            #"Signal":   "time_in_bed"
                    }
                    all_df.loc[len(all_df)] = new_clock_row 
                    
                    
                elif dataset == CHAT:
                    # visit number: 3-> baseline, 10-> followup, only these two, so we couldn't use nonrandomized
                    # start bed: nsrr_begtimbd_f1, need to check if this is the start time of edf
                    # only baseline and followup have comprehensive annotations -> no use of non-randomized split
                    def extract_id_chat(path: str) -> str:
                        stem = Path(path).stem                     # 'chat-followup-300464.edf'
                        return stem.split('-')[2], stem.split('-')[1]                  # '200728'

                    start_time, duration, clock_obj = _extract_start_duration_clock(label_fp)
                    subject_id, visit_id = extract_id_chat(label_fp)
                    if visit_id == 'followup':
                        visit_id = 10
                    elif visit_id == 'baseline':
                        visit_id = 3
                    df = pd.read_csv(MASTER_CHAT[0], encoding="cp1252")   
                    start_sleep_clock = df.loc[
                        (df['nsrrid'] == int(subject_id)) & (df['vnum'] == visit_id),
                        'nsrr_begtimbd_f1'
                    ].iloc[0] # clock time
                    end_sleep_clock =df.loc[
                        (df['nsrrid'] == int(subject_id)) & (df['vnum'] == visit_id),
                        'nsrr_endtimbd_f1'
                    ].iloc[0] # clock time
                    
                    start_sleep_time = (
                            datetime.datetime.strptime(start_sleep_clock, "%H:%M:%S")
                            - datetime.datetime.strptime(clock_obj, "%H:%M:%S")
                        ).total_seconds()
                    end_sleep_time = (
                            datetime.datetime.strptime(end_sleep_clock, "%H:%M:%S")
                            - datetime.datetime.strptime(clock_obj, "%H:%M:%S")
                        ).total_seconds()
                    
                    if start_sleep_time <0:
                        start_sleep_time = start_sleep_time + 86400
                    if end_sleep_time <0:
                        end_sleep_time = end_sleep_time + 86400    
                        
                    new_row = {
                            #"Types":    "time_in_bed",
                            EVENT_NAME_COLUMN: "time_in_bed",
                            START_TIME_COLUMN:   start_sleep_time,
                            END_TIME_COLUMN:     end_sleep_time,
                            #"Signal":   "time_in_bed"
                    }
                    all_df.loc[len(all_df)] = new_row 
                    new_clock_row = {
                            #"Types":    "time_in_bed",
                            EVENT_NAME_COLUMN: "clock_in_bed",
                            START_TIME_COLUMN:   start_sleep_clock,
                            END_TIME_COLUMN:     end_sleep_clock,
                            #"Signal":   "time_in_bed"
                    }
                    all_df.loc[len(all_df)] = new_clock_row 
                    print(all_df)
                    
                elif dataset == MROS:
                    def extract_id_mros(path: str) -> str:
                        stem = Path(path).stem                     
                        return stem.split('-')[2], stem.split('-')[1]                  # '200728'

                    start_time, duration, clock_obj = _extract_start_duration_clock(label_fp)
                    subject_id, visit_id = extract_id_mros(label_fp)
                    subject_id = subject_id.upper()
                    if visit_id == 'visit1':
                        df = pd.read_csv(RAW_MROS[0], encoding="cp1252") 
                    elif visit_id == 'visit2':
                        df = pd.read_csv(RAW_MROS[1], encoding="cp1252") 
                    start_sleep_clock = df.loc[
                        (df['nsrrid'] == subject_id),
                        'postlotp'
                    ].iloc[0] # clock time
                    start_sleep_time = (
                            datetime.datetime.strptime(start_sleep_clock, "%H:%M:%S")
                            - datetime.datetime.strptime(clock_obj, "%H:%M:%S")
                        ).total_seconds()
#                     end_sleep_clock =df.loc[
#                         (df['nsrrid'] == subject_id),
#                         'postendp'
#                     ].iloc[0] # clock time
#                     end_sleep_time = (
#                             datetime.datetime.strptime(end_sleep_clock, "%H:%M:%S")
#                             - datetime.datetime.strptime(clock_obj, "%H:%M:%S")
#                         ).total_seconds()
                    # print(clock_obj, start_sleep_clock)
                    sleep_time_min = df.loc[df['nsrrid'] == subject_id, 'potimebd'].iloc[0] # sleep mins
                    end_sleep_time = start_sleep_time + 60 * sleep_time_min
                    
                    
                    if start_sleep_time <0:
                        start_sleep_time = start_sleep_time + 86400
                    if end_sleep_time <0:
                        end_sleep_time = end_sleep_time + 86400 
                    
                    new_row = {
                            #"Types":    "time_in_bed",
                            EVENT_NAME_COLUMN: "time_in_bed",
                            START_TIME_COLUMN:   start_sleep_time,
                            END_TIME_COLUMN:     end_sleep_time,
                            #"Signal":   "time_in_bed"
                    }
                    all_df.loc[len(all_df)] = new_row 
                    start_sleep_clock = str(datetime.timedelta(seconds=start_sleep_time))[-8:]
                    end_sleep_clock = str(datetime.timedelta(seconds=end_sleep_time))[-8:]
                    new_clock_row = {
                            #"Types":    "time_in_bed",
                            EVENT_NAME_COLUMN: "clock_in_bed",
                            START_TIME_COLUMN:   start_sleep_clock,
                            END_TIME_COLUMN:     end_sleep_clock,
                            #"Signal":   "time_in_bed"
                    }
                    all_df.loc[len(all_df)] = new_clock_row 
                    
                    # print(new_row)
                elif dataset == CCSHS:
                    def extract_id_ccshs(path: str) -> str:
                        stem = Path(path).stem                     # 'ccshs-trec-1800741.edf'
                        return stem.split('-')[2]                  # '200728'
                    start_time, duration, clock_obj = _extract_start_duration_clock(label_fp)
                    subject_id = extract_id_ccshs(label_fp)
                    
                    df = pd.read_csv(MASTER_CCSHS[0], encoding="cp1252")   
                    start_sleep_clock = df.loc[
                        (df['nsrrid'] == int(subject_id)),
                        'nsrr_begtimbd_f1'
                    ].iloc[0] # clock time
                    end_sleep_clock =df.loc[
                        (df['nsrrid'] == int(subject_id)),
                        'nsrr_endtimbd_f1'
                    ].iloc[0] # clock time
                    
                    start_sleep_time = (
                            datetime.datetime.strptime(start_sleep_clock, "%H:%M:%S")
                            - datetime.datetime.strptime(clock_obj, "%H:%M:%S")
                        ).total_seconds()
                    end_sleep_time = (
                            datetime.datetime.strptime(end_sleep_clock, "%H:%M:%S")
                            - datetime.datetime.strptime(clock_obj, "%H:%M:%S")
                        ).total_seconds()
                    
                    if start_sleep_time <0:
                        start_sleep_time = start_sleep_time + 86400
                    if end_sleep_time <0:
                        end_sleep_time = end_sleep_time + 86400    
                      
                    new_row = {
                            #"Types":    "time_in_bed",
                            EVENT_NAME_COLUMN: "time_in_bed",
                            START_TIME_COLUMN:   start_sleep_time,
                            END_TIME_COLUMN:     end_sleep_time,
                            #"Signal":   "time_in_bed"
                    }
                    all_df.loc[len(all_df)] = new_row 
                    new_clock_row = {
                            #"Types":    "time_in_bed",
                            EVENT_NAME_COLUMN: "clock_in_bed",
                            START_TIME_COLUMN:   start_sleep_clock,
                            END_TIME_COLUMN:     end_sleep_clock,
                            #"Signal":   "time_in_bed"
                    }
                    all_df.loc[len(all_df)] = new_clock_row 
                    
                    
                elif dataset == CFS:
                    # start: stloutp, maybe > 86400
                    # end: stlonp, maybe < 86400
                    def extract_id_ccshs(path: str) -> str:
                        stem = Path(path).stem                     # 'ccshs-trec-1800741.edf'
                        return stem.split('-')[2]                  # '200728'
                    start_time, duration, clock_obj = _extract_start_duration_clock(label_fp)
                    subject_id = extract_id_ccshs(label_fp)
                    
                    df = pd.read_csv(RAW_CFS[0], encoding="cp1252")   
                    
                    start_sleep_time = df.loc[
                        (df['nsrrid'] == int(subject_id)),
                        'stloutp'
                    ].iloc[0] 
                    end_sleep_time =df.loc[
                        (df['nsrrid'] == int(subject_id)),
                        'stlonp'
                    ].iloc[0] 
                    start_sleep_clock = str(datetime.timedelta(seconds=start_sleep_time))[-8:]
                    end_sleep_clock = str(datetime.timedelta(seconds=end_sleep_time))[-8:]
                    
                    start_sleep_time = (
                            datetime.datetime.strptime(start_sleep_clock, "%H:%M:%S")
                            - datetime.datetime.strptime(clock_obj, "%H:%M:%S")
                        ).total_seconds()
                    end_sleep_time = (
                            datetime.datetime.strptime(end_sleep_clock, "%H:%M:%S")
                            - datetime.datetime.strptime(clock_obj, "%H:%M:%S")
                        ).total_seconds()

                    
                    if start_sleep_time <0:
                        start_sleep_time = start_sleep_time + 86400
                    if end_sleep_time <0:
                        end_sleep_time = end_sleep_time + 86400    
 
                        
                    new_row = {
                            #"Types":    "time_in_bed",
                            EVENT_NAME_COLUMN: "time_in_bed",
                            START_TIME_COLUMN:   start_sleep_time,
                            END_TIME_COLUMN:     end_sleep_time,
                            #"Signal":   "time_in_bed"
                    }
                    all_df.loc[len(all_df)] = new_row 
                    new_clock_row = {
                            #"Types":    "time_in_bed",
                            EVENT_NAME_COLUMN: "clock_in_bed",
                            START_TIME_COLUMN:   start_sleep_clock,
                            END_TIME_COLUMN:     end_sleep_clock,
                            #"Signal":   "time_in_bed"
                    }
                    all_df.loc[len(all_df)] = new_clock_row 
                    
                    
                elif dataset == MESA:
                    def extract_id_mesa(path: str) -> str:
                        stem = Path(path).stem                     # 'ccshs-trec-1800741.edf'
                        return stem.split('-')[2]                  # '200728'
                    start_time, duration, clock_obj = _extract_start_duration_clock(label_fp)
                    subject_id = extract_id_mesa(label_fp)
                    
                    df = pd.read_csv(RAW_MESA[0], encoding="cp1252")   
                    start_sleep_clock = df.loc[
                        (df['mesaid'] == int(subject_id)),
                        'stloutp5'
                    ].iloc[0] # clock time
                    end_sleep_clock =df.loc[
                        (df['mesaid'] == int(subject_id)),
                        'stlonp5'
                    ].iloc[0] # clock time
                    
                    start_sleep_time = (
                            datetime.datetime.strptime(start_sleep_clock, "%H:%M:%S")
                            - datetime.datetime.strptime(clock_obj, "%H:%M:%S")
                        ).total_seconds()
                    end_sleep_time = (
                            datetime.datetime.strptime(end_sleep_clock, "%H:%M:%S")
                            - datetime.datetime.strptime(clock_obj, "%H:%M:%S")
                        ).total_seconds()
                    
                    if start_sleep_time <0:
                        start_sleep_time = start_sleep_time + 86400
                    if end_sleep_time <0:
                        end_sleep_time = end_sleep_time + 86400    
                      
                    new_row = {
                            #"Types":    "time_in_bed",
                            EVENT_NAME_COLUMN: "time_in_bed",
                            START_TIME_COLUMN:   start_sleep_time,
                            END_TIME_COLUMN:     end_sleep_time,
                            #"Signal":   "time_in_bed"
                    }
                    all_df.loc[len(all_df)] = new_row 
                    new_clock_row = {
                            #"Types":    "time_in_bed",
                            EVENT_NAME_COLUMN: "clock_in_bed",
                            START_TIME_COLUMN:   start_sleep_clock,
                            END_TIME_COLUMN:     end_sleep_clock,
                            #"Signal":   "time_in_bed"
                    }
                    all_df.loc[len(all_df)] = new_clock_row 
                    
                    
                elif dataset == SOF:
                    # longer 
                    def extract_id_sof(path: str) -> str:
                        stem = Path(path).stem                     # 'ccshs-trec-1800741.edf'
                        return stem.split('-')[3]                  # '200728'
                    start_time, duration, clock_obj = _extract_start_duration_clock(label_fp)
                    subject_id = extract_id_sof(label_fp)
                    
                    df = pd.read_csv(RAW_SOF[0], encoding="cp1252")   
                    start_sleep_clock = df.loc[
                        (df['sofid'] == int(subject_id)),
                        'stloutp'
                    ].iloc[0] # clock time
                    
                    end_sleep_min =df.loc[
                        (df['sofid'] == int(subject_id)),
                        'timebedp'
                    ].iloc[0] # clock time
                    
                    start_sleep_time = (
                            datetime.datetime.strptime(start_sleep_clock, "%H:%M:%S")
                            - datetime.datetime.strptime(clock_obj, "%H:%M:%S")
                        ).total_seconds()
                    end_sleep_time = start_sleep_time + end_sleep_min * 60
                    print(start_time, clock_obj, start_sleep_clock)
                    if start_sleep_time <0:
                        start_sleep_time = start_sleep_time + 86400
                    if end_sleep_time <0:
                        end_sleep_time = end_sleep_time + 86400    
                      
                    new_row = {
                            #"Types":    "time_in_bed",
                            EVENT_NAME_COLUMN: "time_in_bed",
                            START_TIME_COLUMN:   start_sleep_time,
                            END_TIME_COLUMN:     end_sleep_time,
                            #"Signal":   "time_in_bed"
                    }
                    all_df.loc[len(all_df)] = new_row 
                    end_sleep_clock = str(datetime.timedelta(seconds=end_sleep_time))[-8:]
                    new_clock_row = {
                            #"Types":    "time_in_bed",
                            EVENT_NAME_COLUMN: "clock_in_bed",
                            START_TIME_COLUMN:   start_sleep_clock,
                            END_TIME_COLUMN:     end_sleep_clock,
                            #"Signal":   "time_in_bed"
                    }
                    all_df.loc[len(all_df)] = new_clock_row 
                    
                    

            except Exception as e:
                logger.error(f'Failed to parse: {label_fp}.')
                logger.error(e)
                return False
            
            all_df = normalize_event_concepts(all_df)
            
        # ---

        # --- Process txt sleep stages and annotations ---
        else:
            if dataset == WSC:
                # main for WSC
                
                if label_fp.endswith(".stg.txt"):
                    temp_annot_fp = label_fp.replace('.stg.txt', '.sco.txt')
                    all_df = parse_sco_wsc(temp_annot_fp)
                else:
                    all_df = parse_all_score_wsc(label_fp)
                    
                #########################
                import re
                filepath = edf_fp.replace('.edf', '.log.txt')
                pattern  = re.compile(r"\bLIGHTS?\s+OUT\b", re.I)   
                light_out_times = []
                with open(filepath, encoding="utf-8") as fh:
                    for line in fh:
                        if pattern.search(line):
                            time_str = line.split()[0]
                            light_out_times.append(time_str)
                            
                pattern  = re.compile(r"\bLIGHTS?\s+ON\b", re.I)   
                light_on_times = []
                with open(filepath, encoding="utf-8") as fh:
                    for line in fh:
                        if pattern.search(line):
                            time_str = line.split()[0]
                            light_on_times.append(time_str)
                            
                pattern  = re.compile(r"\bRecording?\s+Started\b", re.I)   
                record_start_times = []
                with open(filepath, encoding="utf-8") as fh:
                    for line in fh:
                        if pattern.search(line):
                            time_str = line.split()[0]
                            record_start_times.append(time_str)
                            
                if len(light_out_times) > 0:
                    start_sleep_clock = light_out_times[0]
                if len(light_on_times) > 0:
                    end_sleep_clock = light_on_times[0]
                if len(record_start_times) > 0:
                    record_start_clock = record_start_times[0]
                print(start_sleep_clock, end_sleep_clock, record_start_clock)
                
                start_sleep_time = (
                            datetime.datetime.strptime(start_sleep_clock, "%H:%M:%S")
                            - datetime.datetime.strptime(record_start_clock, "%H:%M:%S")
                        ).total_seconds()
                end_sleep_time = (
                            datetime.datetime.strptime(end_sleep_clock, "%H:%M:%S")
                            - datetime.datetime.strptime(record_start_clock, "%H:%M:%S")
                        ).total_seconds()
                if start_sleep_time <0:
                    start_sleep_time = start_sleep_time + 86400
                if end_sleep_time <0:
                    end_sleep_time = end_sleep_time + 86400  
                new_row = {
                            #"Types":    "time_in_bed",
                            EVENT_NAME_COLUMN: "time_in_bed",
                            START_TIME_COLUMN:   start_sleep_time,
                            END_TIME_COLUMN:     end_sleep_time,
                            #"Signal":   "time_in_bed"
                    }
                all_df.loc[len(all_df)] = new_row 
                new_clock_row = {
                            #"Types":    "time_in_bed",
                            EVENT_NAME_COLUMN: "clock_in_bed",
                            START_TIME_COLUMN:   start_sleep_clock,
                            END_TIME_COLUMN:     end_sleep_clock,
                            #"Signal":   "time_in_bed"
                    }
                all_df.loc[len(all_df)] = new_clock_row 
                
                ################################
                if not label_fp.endswith(".stg.txt"):
                    temp_stg_fp = label_fp.replace('.allscore.txt', '.stg.txt')
                    labels = parse_txt_annotations_wsc(fp=temp_stg_fp)
                else:
                    labels = parse_txt_annotations_wsc(fp=label_fp)
                
                all_df = normalize_event_concepts(all_df)
                
                # NOTE: If we end up using a dataset with txt annotations, we will want to write another function to parse all other events
                if labels is None:
                    logger.error(f'Failed to parse: {label_fp}.')
                    return False
            
        # ---

    # --- sleep stage labels ---
    # NOTE: we are storing sleep stages using integers. pls see config.py for mapping. 
        # 0 = awake, 1 = light sleep, 2 = deep sleep, 3 = rem sleep, -1 = unknown or unscored


    labels = labels.reindex(TARGET_LABEL_INDEX).fillna(SLEEP_STAGE_UNKNOWN) # these are sleep stage labels
    # Check for N1, N3 or REM presence. (Recordings with just sleep-wake typically use N2 as sole sleep class)
    stage_counts = labels.value_counts()
    if stage_counts.get(1.0) is None and stage_counts.get(3.0) is None and stage_counts.get(4.0) is None: 
        logger.error(f'No N1, N3 or REM in {label_fp}.')
        output_path_sleep = output_fp + '_sleepstage.issues.parquet' # note these are still useful since sleep stages are not strictly necessary for captions
    else:
        output_path_sleep = output_fp + '_sleepstage.parquet'

   
    
    # --- waveform data ---
    edf, start_time = load_edf_data(edf_fp, columns=EDF_COLS, raise_on_missing=False)
    waveform_df, channel_indicator = process_edf(edf)
    
    for col_name in waveform_df.columns.to_list():
        output_path_channel = output_fp + f'_{col_name}.parquet'
        df_temp = waveform_df[[col_name]].copy()
        df_temp.dropna(inplace = True)
        output_path_channel = output_path_channel.replace('.parquet', '.npz')
        np.savez_compressed(output_path_channel,
                    values=df_temp.values,
                    index=df_temp.index.values)
                    # columns=df_temp.columns.values)
     # labels.to_frame().to_parquet(output_path_sleep)
    output_path_sleep = output_path_sleep.replace('.parquet', '.npz')
    np.savez_compressed(output_path_sleep,
                    values=labels.values,
                    index=labels.index.values)
                    #columns=labels.columns.values)
    # ---

    # --- other events ---
    output_path_events = output_fp + '_events.parquet'
    output_path_events = output_path_events.replace('.parquet', '.npz')
    np.savez_compressed(output_path_events,
                    values=all_df.values,
                    index=all_df.index.values)
                    #columns=all_df.columns.values)

    # ---


    # --- if POS in annotations instead of edf (HSP), indicate presence in channel indicator ---
    if dataset == HSP:
        if body_position_series is not None:
            channel_indicator[POS] = 1
    # ---

    # --- channel indicator ---
    channel_indicator_df = pd.DataFrame([channel_indicator])
    session_id = os.path.basename(output_fp)
    channel_indicator_df.insert(0, 'ID', session_id)
    channel_indicator_df.insert(1, 'START_TOD', start_time)
    channel_indicator_df.to_csv(output_fp + '_channel_indicator.csv', index=False)
    # ---

    # --- HSP download cleanup ---
    if dataset == HSP:
        # IMPORTANT: this is patient - session specific, so it should not interefere with other parallel processes
        # NOTE: might be more efficient to change this to a scratch folder for the job when submitting to slurm 
        shutil.rmtree(download_folder)
    # ---
    
    return True


############### this two functions are related to the file structure!!!!!!!!! check this please!!!!!!!!!!!!!!
def get_edf_path(session_id: str, dataset: str, folder: str):
    if dataset == SHHS:
        partition, _ = session_id.split('-')  # shhs1 or shhs2
        edf_fp = os.path.join(folder, 'polysomnography/edfs', partition, f'{session_id}.edf')
    elif dataset == MROS:
        _, partition, *_ = session_id.split('-')  # mros visit 1 or 2
        edf_fp = os.path.join(folder, 'polysomnography/edfs', partition, f'{session_id}.edf')
    elif dataset == CHAT:
        if 'nonrandomized' in session_id:  # e.g. chat-baseline-nonrandomized-xxxx
            partition = 'nonrandomized'
        else:
            partition = session_id.split('-')[1]  # e.g. chat-baseline-xxxx
        edf_fp = os.path.join(folder, 'polysomnography/edfs', partition, f'{session_id}.edf')
        fixed_edf_fp = edf_fp.replace('.edf', '_fixed.edf')
        # Check for existence of fixed EDF (physical maximum is 0.0 in some files and needed fixing.)
        if os.path.exists(fixed_edf_fp):
            edf_fp = fixed_edf_fp
    else:
        if dataset == CCSHS or dataset == CFS or dataset == MESA or dataset == SOF:
            if 'nonrandomized' in session_id:  # e.g. chat-baseline-nonrandomized-xxxx
                partition = 'nonrandomized'
            else:
                partition = session_id.split('-')[1]  # e.g. chat-baseline-xxxx
            edf_fp = os.path.join(folder, 'polysomnography/edfs', f'{session_id}.edf')
            fixed_edf_fp = edf_fp.replace('.edf', '_fixed.edf')
            # Check for existence of fixed EDF (physical maximum is 0.0 in some files and needed fixing.)
            if os.path.exists(fixed_edf_fp):
                edf_fp = fixed_edf_fp
        else:
            # edf_fp = os.path.join(folder, 'polysomnography/edfs', f'{session_id}.edf')
            if 'nonrandomized' in session_id:  # e.g. chat-baseline-nonrandomized-xxxx
                partition = 'nonrandomized'
            else:
                partition = session_id.split('-')[1]  # e.g. chat-baseline-xxxx
            edf_fp = os.path.join(folder, 'polysomnography/edfs', partition, f'{session_id}.edf')
            fixed_edf_fp = edf_fp.replace('.edf', '_fixed.edf')
            # Check for existence of fixed EDF (physical maximum is 0.0 in some files and needed fixing.)
            if os.path.exists(fixed_edf_fp):
                edf_fp = fixed_edf_fp
    return edf_fp


def prepare_dataset(folder: str, output_folder: str, dataset: str):
    """Prepare dataset IO locations for parallel processing."""
    # WSC uses .txt annotation files
    fp_dict = {}
    if dataset == WSC:
        edf_fps = glob(f'{folder}/polysomnography/*.edf', recursive=True)
        label_fps = []
        all_count = 0
        has_log_count = 0
        for edf_fp in edf_fps:
            all_count +=1
            all_score_fp = edf_fp.replace('.edf', '.allscore.txt')
            stg_fp = edf_fp.replace('.edf', '.stg.txt')
            sco_fp = edf_fp.replace('.edf', '.sco.txt')
            log_fp = edf_fp.replace('.edf', '.log.txt')
            if not os.path.exists(log_fp):
                continue
            has_log_count +=1
            if os.path.exists(all_score_fp):
                label_fp = all_score_fp
                if not os.path.exists(stg_fp):
                    continue
            elif os.path.exists(stg_fp):
                
                label_fp = stg_fp
                if not os.path.exists(sco_fp):
                    continue
            else:
                continue
            session_id = os.path.basename(edf_fp).replace('.edf', '')
            output_fp = os.path.join(output_folder, dataset, INGEST, f'{session_id}')
            fp_dict[session_id] = {'edf_fp': edf_fp, 'label_fp': label_fp, 'output_fp': output_fp}
        print(all_count, has_log_count)
        return fp_dict
    
    # Other datasets use NSRR standardized XML files

    if dataset == CCSHS or dataset == CFS or dataset == MESA or dataset == SOF:
        label_fps = glob(f'{folder}/polysomnography/annotations-events-nsrr/**.xml', recursive=True)
        for label_fp in label_fps:
            session_id = os.path.basename(label_fp).replace('-nsrr.xml', '')
            edf_fp = get_edf_path(session_id, dataset, folder)
            if not os.path.exists(edf_fp):
                logger.warning(f"{edf_fp=} doesn't exist. Skipping...")
                continue
            output_fp = os.path.join(output_folder, dataset, INGEST, f'{session_id}')
            fp_dict[session_id] = {'edf_fp': edf_fp, 'label_fp': label_fp, 'output_fp': output_fp}
    else:
        
        label_fps = glob(f'{folder}/polysomnography/annotations-events-nsrr/**/**.xml', recursive=True)
        count = 0
        for label_fp in label_fps:
            count +=1
            print(count)
            session_id = os.path.basename(label_fp).replace('-nsrr.xml', '')
            
            edf_fp = get_edf_path(session_id, dataset, folder)
            if not os.path.exists(edf_fp):
                logger.warning(f"{edf_fp=} doesn't exist. Skipping...")
                continue
            output_fp = os.path.join(output_folder, dataset, INGEST, f'{session_id}')
            fp_dict[session_id] = {'edf_fp': edf_fp, 'label_fp': label_fp, 'output_fp': output_fp}

    return fp_dict
############### this two functions are related to the file structure!!!!!!!!! check this please!!!!!!!!!!!!!!

def process_files(
    fp_dict, dataset: str, max_parallel: int = 1, overwrite: bool = False, address: str = 'local'
):
    print(f'Preparing to process {len(fp_dict)} files.')

    def proc(arg_dict):
        try:
            return process(overwrite=overwrite, **arg_dict, dataset=dataset)
        except Exception as e:
            logger.error(f'Failed on {arg_dict} - {e}')
            print(f'Failed on {arg_dict} - {e}')
            return False

    if max_parallel > 1:
        MEM_GB = int(os.environ.get('SLURM_MEM_PER_NODE', 8 * 1024)) / 1024  # SLURM env var. is in MB
        ray.init(
            address=address,
            num_cpus=max_parallel,
            object_store_memory=MEM_GB * 1024**3,
            ignore_reinit_error=True,
            include_dashboard=False,
        )
        num_converted = sum(parallelise(proc, fp_dict.values(), use_tqdm=True, max_parallel=max_parallel))
    else:
        num_converted = 0
        for fp_map in tqdm(fp_dict.values()):
            #print(fp_map)
            num_converted += process(**fp_map, dataset=dataset, overwrite=overwrite)
    print(f'Converted {num_converted} files.')


def parse_args():
    parser = argparse.ArgumentParser(prog='Dataset Processor', description='Process dataset.')
    parser.add_argument('--folder', help='Location of dataset.')
    parser.add_argument('--site', default=None, help=f'site number for hsp dataset {VALID_SITES}' )
    parser.add_argument('--max-parallel', default=1, type=int, help='Parallel processes.')
    parser.add_argument('--cluster-address', default='local', type=str, help='Ray cluster address (defaults to local).')
    parser.add_argument('--output-folder', required=True, help='Base output folder for processed datasets.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing parquet files.',
        default=False,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.folder.split('/')[-1]  # Infer name of dataset e.g. path/to/mros
    print(f'Processing {dataset=}...')
    if dataset == HSP:
        hsp_checkio(args) # make sure args.site is valid
        print(f'Processing site {args.site}...')
        fp_dict = prepare_hsp_dataset(folder=args.folder, output_folder=args.output_folder, dataset=dataset, site=args.site)
        """
        For compatibility with the process_files function, I use the same fp_dict keys, but the paths stored in the keys are different from the output of prepare_dataset function. 
        - edf_fp: directory to store the temporary downloads from aws server. this download folder will be deleted after processing each file. 
        - label_fp: the suffix of the aws file path to the patient-level subfolder we want to download. 
        - output_fp: file path of the final processing output
        """
    else:
        print("start preparing files")
        fp_dict = prepare_dataset(folder=args.folder, output_folder=args.output_folder, dataset=dataset)
        print("finish preparing files")
    process_files(fp_dict, dataset, max_parallel=args.max_parallel, overwrite=args.overwrite, address=args.cluster_address)


if __name__ == '__main__':
    main()
