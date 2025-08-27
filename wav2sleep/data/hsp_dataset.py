### this file contains functions and logic specific to HSP dataset processing 

import pandas as pd 
import numpy as np
import os 
import subprocess
from ..settings import *
from ..config import * # imports mapping and inverse mapping dictionaries for annotations
from pathlib import Path
import datetime
import mne
import math
from typing import Union

# --- init any maps to use 
# invert POSITION_MAPPING['HSP'] to map raw event → standardized label
POSITION_MAPPING_INVERSE_HSP = {
    raw_event: standard_label
    for standard_label, raw_list in POSITION_MAPPING[HSP].items()
    for raw_event in raw_list
}

# NOTE: for hsp I0004, periodic leg movement events were named 'right leg', 'both leg', 'left leg'
# it is possible that other datasets use similar naming convention for ISOLATED leg movement events 
# to avoid any bugs for other datasets, I decided to hard-code the event name convergions in the hsp_dataset.py file
LIMB_MOVEMENT_MAPPING_HSP = {
        LIMB_MOVEMENT_ISOLATED: (
        ),
        LIMB_MOVEMENT_PERIODIC: (
            'both leg', 
        ),
        LIMB_MOVEMENT_ISOLATED_LEFT: (
        ),
        LIMB_MOVEMENT_ISOLATED_RIGHT: (
        ),
        LIMB_MOVEMENT_PERIODIC_LEFT: (
            'left leg', 
        ),
        LIMB_MOVEMENT_PERIODIC_RIGHT: (
            'right leg',
        ),
    }

# Storing some annotations potentially useful for QC
BADDATA_WARNING = "Bad Data Warning"
QC_MAPPING_HSP = {
    BADDATA_WARNING: (
        'BadData.evt',
        'baddata.evt',
    ),
}


# --- 

def hsp_checkio(args):
    if args.site is None:
        raise ValueError('site number is required for hsp dataset')
    elif args.site not in VALID_SITES:
        raise ValueError(f'site number is not valid for hsp dataset, valid sites are {VALID_SITES}')
    return

"""
HSP requires sequential download so we need to use the metadata file to get the files paths.
Each site number has different file structure. Within each site, sometimes the naming of files are inconsistent. 
We need to tackle this with the following steps:

1. Use the metadata file, if available, to get patient-level subfolders of interest. 
2. We will use the same process_files function, but the fp_dict will have different paths.
    - edf_fp: directory to store the temporary downloads from aws server. this download folder will be deleted after processing each file. 
    - label_fp: the suffix of the aws file path to the patient-level subfolder we want to download. 
    - output_fp: file path of the final processing output
3. In parallel processing, we need to do the following:
    - get the patient-level subfolder from the fp_dict under 'label_fp' key
    - download the subfolder from the aws server to edf_fp directory, deleting this download directory after each patient (we should do this in scratch space for the  if possible)
    - use the known file suffixes to get each of the files in this , which include edf and annotation files
    - invoke the site-specific processing functions

"""

def prepare_hsp_dataset(folder: str, output_folder: str, dataset: str, site: str):
    """
    Prepare HSP dataset individual-level folder paths.
    """

    print(f'Preparing file paths for site {site}...')

    """
        Desired file structure:

        output_dir/
            - {site}_master.csv # master csv, filtered to files of interest
            - DOWNLOAD/ # location to temporarily store downloaded patient-level folders
                - subfolder/ 
                    - session/
                        - edf file, annotation files
            - INGEST/ # location to store output of preprocessing
                - subfolder-session.*
    """

    fp_dict = {}

    # --- dataset specific logic here only is for relevant patient filtering ---

    if site == S0001:

        metadata_fp = os.path.join(folder, 'psg-metadata', 'I0001_psg_metadata_2025-05-06.csv')
        metadata_df = pd.read_csv(metadata_fp)
        metadata_df_filtered = metadata_df[ (metadata_df.HasSleepAnnotations == 'Y') & (metadata_df.HasStaging == 'Y') ].copy()
        # for S0001, BidsFolder needs to be renamed
        metadata_df_filtered = metadata_df_filtered.rename(columns={'BidsFolder': 'BIDSFolder'})

    if site == I0002:
        metadata_fp = os.path.join(folder, 'psg-metadata', 'I0002_psg_metadata_2025-05-06.csv')
        metadata_df = pd.read_csv(metadata_fp)
        metadata_df_filtered = metadata_df[ (metadata_df.SleepAnnotations == 'Y') & (metadata_df.EventAnnotations == 'Y') & (metadata_df.HasStaging == 'Y') ].copy()

    if site == I0003:
        metadata_fp = os.path.join(folder, 'psg-metadata', 'I0003_psg_metadata_2025-05-06.csv')
        metadata_df = pd.read_csv(metadata_fp)
        metadata_df_filtered = metadata_df[ (metadata_df.HasAnnotations == 'Y') ].copy()

    if site == I0004:
        metadata_fp = os.path.join(folder, 'psg-metadata', 'I0004_psg_metadata_2025-05-06.csv')
        metadata_df = pd.read_csv(metadata_fp)
        metadata_df_filtered = metadata_df[ (metadata_df.SleepAnnotations == 'Y') ].copy()
    
    if site == I0006:
        metadata_fp = os.path.join(folder, 'psg-metadata', 'subject_session_map_I0006.csv')
        metadata_df = pd.read_csv(metadata_fp)
        metadata_df_filtered = metadata_df # no filtering can be done for I0006, only subfolder and session paths
    # ---

    output_dir = os.path.join(output_folder, dataset, site) # for master.csv, folder download and final output
    if not os.path.exists(output_dir): # make the directory if it doesn't exist already
        os.makedirs(output_dir)
    metadata_df_filtered.to_csv(os.path.join(output_dir, f'{site}_master.csv'), index = False, header = True) # filtered metadata file 
    for _, row in metadata_df_filtered.iterrows():
        ID = str(row.BDSPPatientID)
        subfolder = row.BIDSFolder
        session = str(row.SessionID)
        patient_folder = os.path.join(site, subfolder, 'ses-' + session, 'eeg') # folder to download from aws server: os.path.join('blah/blah/bids', patient_folder)
        download_dir = os.path.join(output_dir, 'DOWNLOAD', subfolder, 'ses-' + session) # location to temporarily store patient-level downloaded files
        output_fp = os.path.join(output_dir, INGEST, subfolder + '_ses-' + session)
        fp_dict[ID+'_'+session] = {'edf_fp': download_dir, 'label_fp': patient_folder, 'output_fp': output_fp}

    return fp_dict


def download_hsp_subfolder(aws_folder: str, download_folder: str):
    """
    Download a patient-level subfolder from the AWS server using the AWS CLI.
    
    Args:
        aws_folder (str): The subfolder path in the S3 bucket.
        download_folder (str): Local destination directory.
    """

    s3_uri = f"s3://arn:aws:s3:us-east-1:184438910517:accesspoint/bdsp-psg-access-point/PSG/bids/{aws_folder}"

    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        try:
            subprocess.run([
                "aws", "s3", "cp", s3_uri, download_folder,
                "--recursive"
            ], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error occurred during AWS download: {e}")
            if attempt == max_attempts:
                return False
            else:
                continue


def annotations_preprocess(annotations, fs, t0=None):
    """
    adapted from: https://github.com/bdsp-core/bdsp-sleep-data/blob/main/bdsp_sleep_functions.py

    This function assumes column names consistent with HSP S0001 site, containing at least: ['time',  'duration', 'event']

    input: dataframe annotations.csv
    fs: sampling rate
    t0: start datetime from signal file, if None, from the first line of annotations
        for getting '%Y-%m-%d ' and getting seconds since start of edf file
    output: dataframe annotations with new columns: event starts/ends in seconds and ends
    """
    # set missing duration
    annotations['event'] = annotations.event.astype(str)
    annotations.loc[pd.isna(annotations.duration), 'duration'] = 1/fs # set missing duration to 1/sampling rate
    
    # deal with missing time
    annotations = annotations[pd.notna(annotations.time)].reset_index(drop=True)
    
    # remove negative epochs (these usually corresponding to irrelevant labels before start of edf recording)
    if 'epoch' in annotations.columns: # no epoch info for other sites
        if any(annotations.epoch < 1):
            annotations = annotations[annotations.epoch >= 1].reset_index(drop=True)
            
    # boolean to check if it has date
    has_date = annotations['time'].astype(str).str.contains(r'^\d{4}[-/]\d{2}[-/]\d{2}').all()

    # ensure it is pd.to_datetime
    annotations['time'] = pd.to_datetime(annotations['time']).dt.tz_localize(None)
    
    # check if we need to and can add date and adjust for midnight
    if t0 is None:
        t0 = annotations.time.iloc[0]
        if not has_date:
            raise ValueError('time column does not have date information. Cannot deal with midnight without t0 arg')
    else:
        t0 = pd.to_datetime(t0).tz_localize(None)
        if not has_date:
            print("Annotations do not have date information in time column. Getting date from t0 args. Dealing with midnight.")
            # handles occasional fractional seconds by removing fractional sec part
            annotations['time'] = pd.to_datetime(t0.strftime('%Y-%m-%d ')+annotations['time'].astype(str).str.split('.').str[0], format='%Y-%m-%d %H:%M:%S')
            # deal with midnight
            # assumes time ascending order
            midnight_idx = np.where((annotations.time.dt.hour.values[:-1]==23)&(annotations.time.dt.hour.values[1:]==0))[0]
            if len(midnight_idx)==1:
                annotations.loc[midnight_idx[0]+1:, 'time'] += datetime.timedelta(seconds=86400)
            elif len(midnight_idx)>1:
                raise ValueError('More than 1 midnight point?')

    # subset to events in recording period
    annotations = annotations.loc[annotations.time >= t0]

    # add columns
    annotations['start_sec'] = (annotations['time'] - t0).dt.total_seconds()
    annotations['end_sec'] = annotations['start_sec'] + annotations['duration']
    annotations['idx_start'] = np.round(annotations['start_sec'] * fs).astype(int)
    annotations['idx_end'] = annotations['idx_start'] + np.round(annotations['duration'] * fs).astype(int)

    return annotations

"""
Below, there are two categories of functions to extract annotations:
1. Those that use key word presence in event column to extract standardized set of events. This strategy is used when the event names are free-text and are not already standardized (S0001).
    - For this class of HSP sites, I tested the key word based filtering in a separate jupyter notebook. I extracted all unique event names in a sample of 200-300 nights, and manually checked which key words will select the relevant events. 
2. Those that use direct matching of event names using the EVENT_NAME_MAPPING  in configs. This strategy is used when the event names are already standardized and we just have to rename them (I0004).
    - The consistency of event names in each site is double checked in a jupyter notebook, on a sample of 200-300 nights. 
"""

# --- START OF KEY WORD BASED ANNOTATION EXTRACTION ---
def get_sleep_stage(annotations_df, annotation_indicator_dict):
    """
    Extract sleep stage labels from annotations dataframe.
    """
    # extract sleep stage labels 
    sleep_stages_events = annotations_df.loc[annotations_df.event.apply(lambda x: 'sleep_stage' in str(x).lower()), :].copy()
    if len(sleep_stages_events) == 0:
        annotation_indicator_dict['Sleep Stages'] = False
        return None, annotation_indicator_dict
    else:
        annotation_indicator_dict['Sleep Stages'] = True
    # Initialize the new sleep stage column
    sleep_stages_events['Sleep Stages'] = None
    # Apply labels based on string matching in the 'event' column
    sleep_stages_events.loc[sleep_stages_events.event.str.contains('?', case=False, na=False, regex=False), 'Sleep Stages'] = SLEEP_STAGE_UNKNOWN
    sleep_stages_events.loc[sleep_stages_events.event.str.contains('1', case=False, na=False), 'Sleep Stages'] = SLEEP_STAGE_LIGHT_SLEEP
    sleep_stages_events.loc[sleep_stages_events.event.str.contains('2', case=False, na=False), 'Sleep Stages'] = SLEEP_STAGE_LIGHT_SLEEP
    sleep_stages_events.loc[sleep_stages_events.event.str.contains('3', case=False, na=False), 'Sleep Stages'] = SLEEP_STAGE_DEEP_SLEEP
    sleep_stages_events.loc[sleep_stages_events.event.str.contains('r', case=False, na=False), 'Sleep Stages'] = SLEEP_STAGE_REM_SLEEP
    sleep_stages_events.loc[sleep_stages_events.event.str.contains('w', case=False, na=False), 'Sleep Stages'] = SLEEP_STAGE_WAKE
    sleep_stages_events = sleep_stages_events.sort_values(by='end_sec')
    sleep_stage_series = sleep_stages_events.set_index('end_sec')['Sleep Stages'] # consistent with the rest of the pipeline, make sure this describes the previous 30 seconds
    return sleep_stage_series, annotation_indicator_dict

def get_respiratory_events(annotations_df, annotation_indicator_dict):
    """
    Extract respiratory events from annotations dataframe.
    """
    central_apnea_events = annotations_df.loc[annotations_df.event.str.lower().apply(lambda x: (('central' in x) & ('apnea' in x))), :].copy()
    if len(central_apnea_events) == 0:
        annotation_indicator_dict[RESPIRATORY_EVENT_CENTRAL_APNEA] = False
    else:
        central_apnea_events['event'] = RESPIRATORY_EVENT_CENTRAL_APNEA
        annotation_indicator_dict[RESPIRATORY_EVENT_CENTRAL_APNEA] = True
    obstructive_apnea_events = annotations_df.loc[annotations_df.event.str.lower().apply(lambda x: (('obstructive' in x) & ('apnea' in x))), :].copy()
    if len(obstructive_apnea_events) == 0:
        annotation_indicator_dict[RESPIRATORY_EVENT_OBSTRUCTIVE_APNEA] = False
    else:
        obstructive_apnea_events['event'] = RESPIRATORY_EVENT_OBSTRUCTIVE_APNEA
        annotation_indicator_dict[RESPIRATORY_EVENT_OBSTRUCTIVE_APNEA] = True
    mixed_apnea_events = annotations_df.loc[annotations_df.event.str.lower().apply(lambda x: (('mixed' in x) & ('apnea' in x))), :].copy()
    if len(mixed_apnea_events) == 0:
        annotation_indicator_dict[RESPIRATORY_EVENT_MIXED_APNEA] = False
    else:
        mixed_apnea_events['event'] = RESPIRATORY_EVENT_MIXED_APNEA
        annotation_indicator_dict[RESPIRATORY_EVENT_MIXED_APNEA] = True
    hypopnea_events = annotations_df.loc[annotations_df.event.str.lower().apply(lambda x: ('hypopnea' in x)), :].copy()
    if len(hypopnea_events) == 0:
        annotation_indicator_dict[RESPIRATORY_EVENT_HYPOPNEA] = False
    else:
        hypopnea_events['event'] = RESPIRATORY_EVENT_HYPOPNEA
        annotation_indicator_dict[RESPIRATORY_EVENT_HYPOPNEA] = True
    desaturation_events = annotations_df.loc[annotations_df.event.str.lower().apply(lambda x: ('desaturation' in x)), :].copy()
    if len(desaturation_events) == 0:
        annotation_indicator_dict[RESPIRATORY_EVENT_DESATURATION] = False
    else:
        desaturation_events['event'] = RESPIRATORY_EVENT_DESATURATION
        annotation_indicator_dict[RESPIRATORY_EVENT_DESATURATION] = True
    respiratory_events = pd.concat([central_apnea_events, obstructive_apnea_events, mixed_apnea_events, hypopnea_events, desaturation_events])
    #respiratory_events = respiratory_events.sort_values(by='start_sec')
    return respiratory_events, annotation_indicator_dict

def get_limb_events(annotations_df, annotation_indicator_dict):
    """
    Extract limb events from annotations dataframe.
    """
    limb_isolated_events = annotations_df.loc[annotations_df.event.str.lower().apply(lambda x: (('plm' in x) | ('limb' in x)) & ('isolated' in x) ), :].copy()
    if len(limb_isolated_events) == 0:
        annotation_indicator_dict[LIMB_MOVEMENT_ISOLATED] = False
    else:
        limb_isolated_events['event'] = LIMB_MOVEMENT_ISOLATED
        annotation_indicator_dict[LIMB_MOVEMENT_ISOLATED] = True
    limb_periodic_events = annotations_df.loc[annotations_df.event.str.lower().apply(lambda x: (('plm' in x) | ('limb' in x)) & ('periodic' in x) ), :].copy()
    if len(limb_periodic_events) == 0:
        annotation_indicator_dict[LIMB_MOVEMENT_PERIODIC] = False
    else:
        limb_periodic_events['event'] = LIMB_MOVEMENT_PERIODIC
        annotation_indicator_dict[LIMB_MOVEMENT_PERIODIC] = True
    limb_arousal_events = annotations_df.loc[annotations_df.event.str.lower().apply(lambda x: (('plm' in x) | ('limb' in x)) & ('arousal' in x) ), :].copy()
    if len(limb_arousal_events) == 0:
        annotation_indicator_dict[AROUSAL_EVENT_EMG] = False
    else:
        limb_arousal_events['event'] = AROUSAL_EVENT_EMG
        annotation_indicator_dict[AROUSAL_EVENT_EMG] = True
    limb_events = pd.concat([limb_isolated_events, limb_periodic_events, limb_arousal_events])
    #limb_events = limb_events.sort_values(by='start_sec')
    return limb_events, annotation_indicator_dict

def get_arousal_events(annotations_df, annotation_indicator_dict):
    """
    Extract arousal events from annotations dataframe.
    """
    classic_arousal_events = annotations_df.loc[annotations_df.event.str.lower().apply(lambda x: (('arousal' in x) & ('spontaneous' in x) & ('post' not in x))), :].copy() # ASDA arousal events
    # I found through exploration that the only arousal event that is not respiratory or EMG related is spontaneous arousal, and this fits within the classic arousal event category
    if len(classic_arousal_events) == 0:
        annotation_indicator_dict[AROUSAL_EVENT_CLASSIC] = False
    else:
        classic_arousal_events['event'] = AROUSAL_EVENT_CLASSIC
        annotation_indicator_dict[AROUSAL_EVENT_CLASSIC] = True
    rera_events = annotations_df.loc[annotations_df.event.str.lower().apply(lambda x: (('resp' in x) & ('event' in x) & ('arousal' in x)) | ('rera' in x)), :].copy()
    if len(rera_events) == 0:
        annotation_indicator_dict[AROUSAL_EVENT_RESPIRATORY] = False
    else:
        rera_events['event'] = AROUSAL_EVENT_RESPIRATORY
        annotation_indicator_dict[AROUSAL_EVENT_RESPIRATORY] = True
    arousal_events = pd.concat([classic_arousal_events, rera_events])
    #arousal_events = arousal_events.sort_values(by='start_sec')
    return arousal_events, annotation_indicator_dict

def get_lights_events(annotations_df, annotation_indicator_dict):
    """
    Extract lights on/off events from annotations dataframe.
    """
    lights_off_events = annotations_df[annotations_df['event'].isin(EVENT_NAME_MAPPING['Lights Events'][LIGHTS_OFF])].copy()
    lights_off_events['event'] = LIGHTS_OFF
    lights_on_events = annotations_df[annotations_df['event'].isin(EVENT_NAME_MAPPING['Lights Events'][LIGHTS_ON])].copy()
    lights_on_events['event'] = LIGHTS_ON
    """
    Sometimes lights on comes before lights off because these were not ordered properly in the original annotations file, even if the other rows are ordered properly
    When the lights on event with the correct time stamp comes before midnight, our logic does not add 86400 seconds to that time stamp
    So when lights on comes before lights off, we need to add 86400 seconds to the lights on event. This handles most cases properly.

    However, there are some other cases where the lights on and lights off tie stamps do not make sense in the context of the rest of the recording. 
    For example, first time stamps of the night is 22:00:00 and last one is 06:00:00 but the lights on is 17:00:00 and lights off is 09:00:00. These make no sense. 
    In this case we should drop the lights on/off events. The rest of the annotations look fine
    """

    annotation_indicator_dict[LIGHTS_ON] = True
    annotation_indicator_dict[LIGHTS_OFF] = True
    
    # --- quick QC ---
        # QC flag, set to False if any of the following conditions are met:
        # if either light on or light off is missing, 
        # or if there are more than 1 light on or more than 1 light off, 
        # or if lights on comes before lights off and the resulting lights on start_sec after adjusting for midnight is greater than the last event's start_sec by more than 2 hours
    if len(lights_on_events) != 1:
        annotation_indicator_dict[LIGHTS_ON] = False
        lights_on_events = lights_on_events.drop(lights_on_events.index)
    if len(lights_off_events) != 1:
        annotation_indicator_dict[LIGHTS_OFF] = False
        lights_off_events = lights_off_events.drop(lights_off_events.index)
    if len(lights_on_events) == 1 and len(lights_off_events) == 1:
        if lights_on_events.start_sec.iloc[0] < lights_off_events.start_sec.iloc[0]:
            lights_on_events.loc[:, 'start_sec'] += 86400
            if lights_on_events.start_sec.iloc[0] >= annotations_df.start_sec.iloc[-1] + 2*3600:
                # some edge cases where this causes lights on event to be much after the last event annotation
                # in this case, we should drop the lights on/off events
                lights_on_events = lights_on_events.drop(lights_on_events.index)
                lights_off_events = lights_off_events.drop(lights_off_events.index)
                annotation_indicator_dict[LIGHTS_ON] = False
                annotation_indicator_dict[LIGHTS_OFF] = False
    # --- 
    lights_events = pd.concat([lights_off_events, lights_on_events])
    #lights_events = lights_events.sort_values(by='start_sec')
    return lights_events, annotation_indicator_dict


def get_body_position(annotations_df, recording_len_sec, annotation_indicator_dict):
    """
    Convert annotation events into body position labels per second, assuming body positions are logged only when changed.
    This is consistent with assumptions in https://github.com/bdsp-core/bdsp-sleep-data/blob/main/bdsp_sleep_functions.py
    
    Parameters:
        annotations_df: pd.DataFrame with columns including 'event', 'start_sec'
        recording_len_sec: total duration of the recording in seconds

    Returns:
        pd.Series of body position labels per second
    """

    # filter annotations that are position events
    annotations_position = annotations_df[
        annotations_df['event'].isin(POSITION_MAPPING_INVERSE_HSP.keys())
    ].copy().reset_index(drop=True)

    if len(annotations_position) == 0:
        annotation_indicator_dict['Body Positions'] = False
        return None, annotation_indicator_dict
    else:
        annotation_indicator_dict['Body Positions'] = True

    recording_len_sec = int(recording_len_sec) # make sure it's an integer

    # initialize with default BODY_POSITION_OTHER_UNKNOWN
    body_position = pd.Series(
        data=[BODY_POSITION_OTHER_UNKNOWN] * (recording_len_sec*FREQ_POS),
        index=np.arange(0, recording_len_sec + 1e-9, 1/FREQ_POS)[1:],
        name=POS
    )

    # standardize event names to common labels
    annotations_position['standard_label'] = annotations_position['event'].map(POSITION_MAPPING_INVERSE_HSP)

    # Fill body_position per second
    for jloc, row in annotations_position.iterrows():
        start_sec = int(row['start_sec'])
        if jloc == annotations_position.index[-1]:
            end_sec = int(recording_len_sec + 1)
        else:
            end_sec = int(annotations_position.loc[jloc + 1, 'start_sec'])
        body_position.iloc[start_sec:end_sec] = row['standard_label']

    return body_position, annotation_indicator_dict

def annotations_process(annotations_df: pd.DataFrame, fs: Union[int, float], len_sec: Union[int, float], t0=None):
    """
    Process annotations.
    
    Args:
        annotations_df (pd.DataFrame): annotations dataframe
        fs (int or float): sampling frequency
        len_sec (int or float): length of recording in seconds
        t0: start time of recording from edf file
    """

    # if duration column is not present, return failed processing
        # we can impute that sleep stages are 30 sec and respiratory events are 10 sec but...
        # only happens in 8/600 in S0001, so better to maintain data integrity and drop these files
    if ('duration' not in annotations_df.columns) or (all(pd.isna(annotations_df.duration))):
        return None, None, None, None
           
    # preprocess: set missing duration to 1/fs, drop negative epochs, deal with midnight times stamps, calculate start_sec
    annotations_df = annotations_preprocess(annotations_df, fs, t0=t0)

    # annotation qc dictionary, indicating presence of each event annotation
    annotation_indicator_dict = {}
    
    # extract post-processing relevant events (1) Lights on/off
    lights_events_df, annotation_indicator_dict = get_lights_events(annotations_df, annotation_indicator_dict)
    #recording_resume_df = annotations_df[annotations_df['event'] == 'Recording Resumed'].copy() # from my exploration, these did not seem to be problematic
    # extract arousals 
    arousal_events_df, annotation_indicator_dict = get_arousal_events(annotations_df, annotation_indicator_dict)

    # extract limb movements
    #limb_events = [k for k in events_counter if  ('plm' in str(k).lower()) or ('limb' in str(k).lower()) ]
    limb_events_df, annotation_indicator_dict = get_limb_events(annotations_df, annotation_indicator_dict)

    # extract respiratory events 
    respiratory_events_df, annotation_indicator_dict = get_respiratory_events(annotations_df, annotation_indicator_dict)

    # extract sleep stage labels 
    sleep_stage_series, annotation_indicator_dict = get_sleep_stage(annotations_df, annotation_indicator_dict) # for sleep stage, it's already sorted by start_sec

    # extract body position, per second
    body_position_series, annotation_indicator_dict = get_body_position(annotations_df, len_sec, annotation_indicator_dict)

    # combine all other events into a single dataframe
    all_df = pd.concat([lights_events_df, arousal_events_df, limb_events_df, respiratory_events_df])

    if len(all_df) > 0:

        # sort by start_sec and reset index
        all_df = all_df.sort_values(by='start_sec').reset_index(drop=True)

        # subset to event, start_sec, end_sec (we probably don't need epoch, time, duration, idx_start, idx_end anymore)
        all_df = all_df[['event', 'start_sec', 'end_sec']]

        # rename columns to standard names
        all_df.rename(columns={'event': EVENT_NAME_COLUMN, 'start_sec': START_TIME_COLUMN, 'end_sec': END_TIME_COLUMN}, inplace=True)
    
    else:
        
        all_df = None

    # NOTE: sleep_stage_series is None if sleep stages are missing
    # NOTE: body_position_series is None if body positions are missing
    # NOTE: all_df is None if no other annotation events are present
    # NOTE: all indicated in annotation_indicator_dict

    return sleep_stage_series, body_position_series, all_df, annotation_indicator_dict

# --- END OF KEY WORD BASED ANNOTATION EXTRACTION ---

# --- START OF EXACT MATCHING BASED ANNOTATION EXTRACTION ---
def get_baddata_events_matching(annotations_df: pd.DataFrame, annotation_indicator_dict: dict):
    """
    Extract bad data events from annotations dataframe.
    """
    baddata_events = annotations_df[annotations_df['event'].isin(QC_MAPPING_HSP[BADDATA_WARNING])].copy()
    baddata_events['event'] = BADDATA_WARNING
    if len(baddata_events) > 0:
        annotation_indicator_dict[BADDATA_WARNING] = True
    else:
        annotation_indicator_dict[BADDATA_WARNING] = False
    return baddata_events, annotation_indicator_dict

def get_arousal_events_matching(annotations_df: pd.DataFrame, annotation_indicator_dict: dict):
    """
    Extract arousal events from annotations dataframe.
    """
    arousal_classic_events = annotations_df[annotations_df['event'].isin(EVENT_NAME_MAPPING['Arousal Events'][AROUSAL_EVENT_CLASSIC])].copy()
    arousal_classic_events['event'] = AROUSAL_EVENT_CLASSIC
    if len(arousal_classic_events) > 0:
        annotation_indicator_dict[AROUSAL_EVENT_CLASSIC] = True
    else:
        annotation_indicator_dict[AROUSAL_EVENT_CLASSIC] = False
    rera_events = annotations_df[annotations_df['event'].isin(EVENT_NAME_MAPPING['Arousal Events'][AROUSAL_EVENT_RESPIRATORY])].copy()
    rera_events['event'] = AROUSAL_EVENT_RESPIRATORY
    if len(rera_events) > 0:
        annotation_indicator_dict[AROUSAL_EVENT_RESPIRATORY] = True
    else:
        annotation_indicator_dict[AROUSAL_EVENT_RESPIRATORY] = False
    arousal_emg_events = annotations_df[annotations_df['event'].isin(EVENT_NAME_MAPPING['Arousal Events'][AROUSAL_EVENT_EMG])].copy()
    arousal_emg_events['event'] = AROUSAL_EVENT_EMG
    if len(arousal_emg_events) > 0:
        annotation_indicator_dict[AROUSAL_EVENT_EMG] = True
    else:
        annotation_indicator_dict[AROUSAL_EVENT_EMG] = False
    arousal_events = pd.concat([arousal_classic_events, rera_events, arousal_emg_events])
    return arousal_events, annotation_indicator_dict


def get_limb_events_matching(annotations_df: pd.DataFrame, annotation_indicator_dict: dict):
    """
    Extract limb movement events from annotations dataframe.

    NOTE: LIMB_MOVEMENT_MAPPING_HSP is used to avoid confusion in other datasets. HSP uses very generic event names for PLM such as 'both leg', 'left leg', 'right leg'
    """
    limb_isolated_events = annotations_df[annotations_df['event'].isin(LIMB_MOVEMENT_MAPPING_HSP[LIMB_MOVEMENT_ISOLATED])].copy()
    limb_isolated_events['event'] = LIMB_MOVEMENT_ISOLATED
    if len(limb_isolated_events) > 0:
        annotation_indicator_dict[LIMB_MOVEMENT_ISOLATED] = True
    else:
        annotation_indicator_dict[LIMB_MOVEMENT_ISOLATED] = False
    limb_periodic_events = annotations_df[annotations_df['event'].isin(LIMB_MOVEMENT_MAPPING_HSP[LIMB_MOVEMENT_PERIODIC])].copy()
    limb_periodic_events['event'] = LIMB_MOVEMENT_PERIODIC
    if len(limb_periodic_events) > 0:
        annotation_indicator_dict[LIMB_MOVEMENT_PERIODIC] = True
    else:
        annotation_indicator_dict[LIMB_MOVEMENT_PERIODIC] = False
    limb_isolated_left_events = annotations_df[annotations_df['event'].isin(LIMB_MOVEMENT_MAPPING_HSP[LIMB_MOVEMENT_ISOLATED_LEFT])].copy()
    limb_isolated_left_events['event'] = LIMB_MOVEMENT_ISOLATED_LEFT
    if len(limb_isolated_left_events) > 0:
        annotation_indicator_dict[LIMB_MOVEMENT_ISOLATED_LEFT] = True
    else:
        annotation_indicator_dict[LIMB_MOVEMENT_ISOLATED_LEFT] = False
    limb_isolated_right_events = annotations_df[annotations_df['event'].isin(LIMB_MOVEMENT_MAPPING_HSP[LIMB_MOVEMENT_ISOLATED_RIGHT])].copy()
    limb_isolated_right_events['event'] = LIMB_MOVEMENT_ISOLATED_RIGHT
    if len(limb_isolated_right_events) > 0:
        annotation_indicator_dict[LIMB_MOVEMENT_ISOLATED_RIGHT] = True
    else:
        annotation_indicator_dict[LIMB_MOVEMENT_ISOLATED_RIGHT] = False
    limb_periodic_left_events = annotations_df[annotations_df['event'].isin(LIMB_MOVEMENT_MAPPING_HSP[LIMB_MOVEMENT_PERIODIC_LEFT])].copy()
    limb_periodic_left_events['event'] = LIMB_MOVEMENT_PERIODIC_LEFT
    if len(limb_periodic_left_events) > 0:
        annotation_indicator_dict[LIMB_MOVEMENT_PERIODIC_LEFT] = True
    else:
        annotation_indicator_dict[LIMB_MOVEMENT_PERIODIC_LEFT] = False
    limb_periodic_right_events = annotations_df[annotations_df['event'].isin(LIMB_MOVEMENT_MAPPING_HSP[LIMB_MOVEMENT_PERIODIC_RIGHT])].copy()
    limb_periodic_right_events['event'] = LIMB_MOVEMENT_PERIODIC_RIGHT
    if len(limb_periodic_right_events) > 0:
        annotation_indicator_dict[LIMB_MOVEMENT_PERIODIC_RIGHT] = True
    else:
        annotation_indicator_dict[LIMB_MOVEMENT_PERIODIC_RIGHT] = False
    limb_events = pd.concat([limb_isolated_events, limb_periodic_events, limb_isolated_left_events, limb_isolated_right_events, limb_periodic_left_events, limb_periodic_right_events])
    return limb_events, annotation_indicator_dict

def get_respiratory_events_matching(annotations_df: pd.DataFrame, annotation_indicator_dict: dict):
    """
    Extract respiratory events from annotations dataframe.
    """
    centralapnea_events = annotations_df[annotations_df['event'].isin(EVENT_NAME_MAPPING['Respiratory Events'][RESPIRATORY_EVENT_CENTRAL_APNEA])].copy()
    centralapnea_events['event'] = RESPIRATORY_EVENT_CENTRAL_APNEA
    if len(centralapnea_events) > 0:
        annotation_indicator_dict[RESPIRATORY_EVENT_CENTRAL_APNEA] = True
    else:
        annotation_indicator_dict[RESPIRATORY_EVENT_CENTRAL_APNEA] = False
    obstructiveapnea_events = annotations_df[annotations_df['event'].isin(EVENT_NAME_MAPPING['Respiratory Events'][RESPIRATORY_EVENT_OBSTRUCTIVE_APNEA])].copy()
    obstructiveapnea_events['event'] = RESPIRATORY_EVENT_OBSTRUCTIVE_APNEA
    if len(obstructiveapnea_events) > 0:
        annotation_indicator_dict[RESPIRATORY_EVENT_OBSTRUCTIVE_APNEA] = True
    else:
        annotation_indicator_dict[RESPIRATORY_EVENT_OBSTRUCTIVE_APNEA] = False
    mixedapnea_events = annotations_df[annotations_df['event'].isin(EVENT_NAME_MAPPING['Respiratory Events'][RESPIRATORY_EVENT_MIXED_APNEA])].copy()
    mixedapnea_events['event'] = RESPIRATORY_EVENT_MIXED_APNEA
    if len(mixedapnea_events) > 0:
        annotation_indicator_dict[RESPIRATORY_EVENT_MIXED_APNEA] = True
    else:
        annotation_indicator_dict[RESPIRATORY_EVENT_MIXED_APNEA] = False
    hypopnea_events = annotations_df[annotations_df['event'].isin(EVENT_NAME_MAPPING['Respiratory Events'][RESPIRATORY_EVENT_HYPOPNEA])].copy()
    hypopnea_events['event'] = RESPIRATORY_EVENT_HYPOPNEA
    if len(hypopnea_events) > 0:
        annotation_indicator_dict[RESPIRATORY_EVENT_HYPOPNEA] = True
    else:
        annotation_indicator_dict[RESPIRATORY_EVENT_HYPOPNEA] = False
    desaturation_events = annotations_df[annotations_df['event'].isin(EVENT_NAME_MAPPING['Respiratory Events'][RESPIRATORY_EVENT_DESATURATION])].copy()
    desaturation_events['event'] = RESPIRATORY_EVENT_DESATURATION
    if len(desaturation_events) > 0:
        annotation_indicator_dict[RESPIRATORY_EVENT_DESATURATION] = True
    else:
        annotation_indicator_dict[RESPIRATORY_EVENT_DESATURATION] = False
    respiratory_events = pd.concat([centralapnea_events, obstructiveapnea_events, mixedapnea_events, hypopnea_events, desaturation_events])
    return respiratory_events, annotation_indicator_dict

def get_sleep_stage_matching(annotations_df: pd.DataFrame, annotation_indicator_dict: dict):
    """
    Extract sleep stage events from annotations dataframe.
    """
    sleep_stage_events = annotations_df[annotations_df['event'].isin(SLEEP_STAGE_INVERSE_MAPPING.keys())].copy()
    sleep_stage_events['Sleep Stages'] = sleep_stage_events['event'].map(SLEEP_STAGE_INVERSE_MAPPING)
    if len(sleep_stage_events) > 0:
        annotation_indicator_dict['Sleep Stages'] = True
    else:
        annotation_indicator_dict['Sleep Stages'] = False
        return None, annotation_indicator_dict
    sleep_stage_events = sleep_stage_events.sort_values(by='end_sec')
    sleep_stage_series = sleep_stage_events.set_index('end_sec')['Sleep Stages']
    return sleep_stage_series, annotation_indicator_dict


def annotations_process_matching(annotations_df: pd.DataFrame, fs: Union[int, float], len_sec: Union[int, float], t0=None):
    """
    Process annotations with exact event name matching. 
    This function is used when the event column names are standardized to a limited set, and we can directly map each event name to the event labels we desire

    Args:
        annotations_df (pd.DataFrame): annotations dataframe
        fs (int or float): sampling frequency
        len_sec (int or float): length of recording in seconds
        t0: start time of recording from edf file
    """

    # if duration column is not present, return failed processing
        # we can impute that sleep stages are 30 sec and respiratory events are 10 sec but...
        # only happens in 8/600 in S0001, so better to maintain data integrity and drop these files
    if ('duration' not in annotations_df.columns) or (all(pd.isna(annotations_df.duration))):
        return None, None, None, None
           
    # preprocess: set missing duration to 1/fs, deal with midnight times stamps, calculate start_sec
    annotations_df = annotations_preprocess(annotations_df, fs, t0=t0)

    # annotation qc dictionary, indicating presence of each event annotation
    annotation_indicator_dict = {}

    # LIMB_MOVEMENT_INVERSE_MAPPING_HSP, SLEEP_STAGE_INVERSE_MAPPING, RESPIRATORY_INVERSE_MAPPING, AROUSAL_INVERSE_MAPPING, LIGHTS_INVERSE_MAPPING

    # extract post-processing relevant events (1) Lights on/off and (2) bad data indicators?
    lights_events_df, annotation_indicator_dict = get_lights_events(annotations_df, annotation_indicator_dict) # this function already uses exact matching
    baddata_events_df, annotation_indicator_dict = get_baddata_events_matching(annotations_df, annotation_indicator_dict)

    print('lights_events_df')
    print(lights_events_df)
    print('baddata_events_df')
    print(baddata_events_df)
    
    # extract arousals 
    arousal_events_df, annotation_indicator_dict = get_arousal_events_matching(annotations_df, annotation_indicator_dict)

    print('arousal_events_df')
    print(arousal_events_df)

    # extract limb movements
    #limb_events = [k for k in events_counter if  ('plm' in str(k).lower()) or ('limb' in str(k).lower()) ]
    limb_events_df, annotation_indicator_dict = get_limb_events_matching(annotations_df, annotation_indicator_dict)

    print('limb_events_df')
    print(limb_events_df)

    # extract respiratory events 
    respiratory_events_df, annotation_indicator_dict = get_respiratory_events_matching(annotations_df, annotation_indicator_dict)

    print('respiratory_events_df')  
    print(respiratory_events_df)

    # extract sleep stage labels 
    sleep_stage_series, annotation_indicator_dict = get_sleep_stage_matching(annotations_df, annotation_indicator_dict) # for sleep stage, it's already sorted by start_sec

    # check that the sleep stage times 
    if sleep_stage_series is not None:
        if not all(sleep_stage_series.index % 30 == 0):
            print("Sleep stage timestamps do not align with 30 sec epochs")
            return None, None, None, None

    # extract body position, per second (no position data in I0004)
    body_position_series = None
    #body_position_series, annotation_indicator_dict = get_body_position_matching(annotations_df, len_sec, annotation_indicator_dict)

    # combine all other events into a single dataframe
    all_df = pd.concat([lights_events_df, baddata_events_df, arousal_events_df, limb_events_df, respiratory_events_df])

    if len(all_df) > 0:

        # sort by start_sec and reset index
        all_df = all_df.sort_values(by='start_sec').reset_index(drop=True)

        # subset to event, start_sec, end_sec (we probably don't need epoch, time, duration, idx_start, idx_end anymore)
        all_df = all_df[['event', 'start_sec', 'end_sec']]

        # rename columns to standard names
        all_df.rename(columns={'event': EVENT_NAME_COLUMN, 'start_sec': START_TIME_COLUMN, 'end_sec': END_TIME_COLUMN}, inplace=True)
    
    else:
        
        all_df = None

    # NOTE: sleep_stage_series is None if sleep stages are missing
    # NOTE: body_position_series is None if body positions are missing
    # NOTE: all_df is None if no other annotation events are present
    # NOTE: all indicated in annotation_indicator_dict

    return sleep_stage_series, body_position_series, all_df, annotation_indicator_dict

# --- END OF EXACT MATCHING BASED ANNOTATION EXTRACTION ---


def process_hsp_main(download_folder: str, site: str):
    """
    Main function to process HSP dataset.

    Args:
        download_folder (str): Path to the downloaded patient-level subfolder.
        site (str): Site number.

    Returns:
        edf_fp (str): Path to the edf file.
        labels (pd.DataFrame): Sleep stage labels.
        all_df (pd.DataFrame): All annotations.
        error (str): Error message.
    """

    # --- define path object ---
    path_obj = Path(download_folder)

    if site == S0001:
        # --- get file names ---
        edf_files = [f.name for f in path_obj.iterdir() if ".edf" in f.name.lower()]
        annotation_files = [f.name for f in path_obj.glob("*_annotations.csv")]
        presleep_files = [f.name for f in path_obj.glob("*pre.csv")]
        subfolder = download_folder.split(os.sep)[-2] # get subfolder name from download_folder
        session = download_folder.split(os.sep)[-1].replace('ses-', '') # get session id from download_folder

        # --- define edf file path to return ---
        if len(edf_files) == 1:
            edf_fp = os.path.join(download_folder, edf_files[0])
        else:
            error = f"Expected 1 edf file, found {len(edf_files)}"
            return None, None, None, None, None, None, error

        # --- define annotation file to parse ---
        if len(annotation_files) == 1:
            annotations_df = pd.read_csv(os.path.join(download_folder, annotation_files[0]))
            # --- invoke annotation parsing function
            raw_temp = mne.io.read_raw_edf(edf_fp, preload=False) # preload = False, I just want to get the start time and sampling frequency for annotations processing
            # note that if edf file channels have variable sampling frequencies, raw_temp.info['sfreq'] is the sampling frequency of the largest channel
            # that is fine, and len_sec is still the accurate number of seconds in the recoridng
            # https://mne.tools/stable/generated/mne.io.read_raw_edf.html
            start_time = raw_temp.info['meas_date']
            sfreq = raw_temp.info['sfreq'] # used to fill duration of events when missing, i.e. 1/sfreq
            len_sec = raw_temp.n_times / sfreq # used to get number of seconds in the recording for body position series
            labels, body_position_series, all_df, annotation_indicator_dict = annotations_process(annotations_df, sfreq, len_sec, t0=start_time)
            if labels is None and all_df is None: # if no sleep stages and no other annotation events, dropping
                error = f"Failed to process annotations for {download_folder}, annotation files are missing event durations or both sleep stages and event annotations are missing"
                return None, None, None, None, None, None, error
        else:
            error = f"Expected 1 annotation file, found {len(annotation_files)}"
            return None, None, None, None, None, None, error

        # --- define presleep file to parse ---
        # read master csv to merge presleep data into it
        presleep_df_filtered = None
        if len(presleep_files) == 1: # for some people, presleep questionnaire is recorded
            presleep_df = pd.read_csv(os.path.join(download_folder, presleep_files[0]), header = None)
            # extract HSP_S0001_CLINICAL_VARS and save as a csv file
            presleep_df = presleep_df.set_index(0, drop=True).iloc[1:]
            presleep_df_filtered = presleep_df.loc[HSP_S0001_CLINICAL_VARS]  
            presleep_df_filtered = presleep_df_filtered.T
            presleep_df_filtered.insert(0, 'BidsFolder', subfolder)
            presleep_df_filtered.insert(1, 'SessionID', session)
            annotation_indicator_dict['presleep_questionnaire'] = True
        else:
            annotation_indicator_dict['presleep_questionnaire'] = False
        # ---

        # --- turn annotations_indicator into dataframe --- 
        annotation_indicator_df = pd.DataFrame([annotation_indicator_dict])
        annotation_indicator_df.insert(0, 'BidsFolder', subfolder)
        annotation_indicator_df.insert(1, 'SessionID', session)

        # --- return ---
        return edf_fp, labels, body_position_series, all_df, presleep_df_filtered, annotation_indicator_df, None
    
    if site == I0004:
        # --- get file names ---
        edf_files = [f.name for f in path_obj.iterdir() if ".edf" in f.name.lower()]
        annotation_files = [f.name for f in path_obj.glob("*_annotations.csv")]
        subfolder = download_folder.split(os.sep)[-2] # get subfolder name from download_folder
        session = download_folder.split(os.sep)[-1].replace('ses-', '') # get session id from download_folder

        # --- define edf file path to return ---
        if len(edf_files) == 1:
            edf_fp = os.path.join(download_folder, edf_files[0])
        else:
            error = f"Expected 1 edf file, found {len(edf_files)}"
            return None, None, None, None, None, None, error

        # --- define annotation file to parse ---
        if len(annotation_files) == 1:
            annotations_df = pd.read_csv(os.path.join(download_folder, annotation_files[0]))
            # --- invoke annotation parsing function
            raw_temp = mne.io.read_raw_edf(edf_fp, preload=False) # preload = False, I just want to get the start time and sampling frequency for annotations processing
            # note that if edf file channels have variable sampling frequencies, raw_temp.info['sfreq'] is the sampling frequency of the largest channel
            # that is fine, and len_sec is still the accurate number of seconds in the recoridng
            # https://mne.tools/stable/generated/mne.io.read_raw_edf.html
            start_time = raw_temp.info['meas_date']
            sfreq = raw_temp.info['sfreq'] # used to fill duration of events when missing, i.e. 1/sfreq
            len_sec = raw_temp.n_times / sfreq # used to get number of seconds in the recording for body position series
            # rename columns to S0001 names
            annotations_df = annotations_df.rename(columns={
                                                    'Start Time': 'time',
                                                    'Duration (seconds)': 'duration', 
                                                    'Event': 'event'
                                                })
            # for rows with Source == 'BadData.evt' or Source == 'baddata.evt', replace event column with Source column 
            baddata_events_bool = annotations_df['Source'].isin(['BadData.evt', 'baddata.evt'])
            annotations_df.loc[baddata_events_bool, 'event'] = annotations_df.loc[baddata_events_bool, 'Source']
            # invoke exact matching annotation processing function 
            labels, body_position_series, all_df, annotation_indicator_dict = annotations_process_matching(annotations_df, sfreq, len_sec, t0=start_time)
            if labels is None and all_df is None: # if no sleep stages and no other annotation events, dropping
                error = f"Failed to process annotations for {download_folder}, annotation files are missing event durations or both sleep stages and event annotations are missing"
                return None, None, None, None, None, None, error
        else:
            error = f"Expected 1 annotation file, found {len(annotation_files)}"
            return None, None, None, None, None, None, error

        # --- turn annotations_indicator into dataframe --- 
        annotation_indicator_df = pd.DataFrame([annotation_indicator_dict])
        annotation_indicator_df.insert(0, 'BidsFolder', subfolder)
        annotation_indicator_df.insert(1, 'SessionID', session)

        # --- return ---
        return edf_fp, labels, body_position_series, all_df, None, annotation_indicator_df, None

    



        

