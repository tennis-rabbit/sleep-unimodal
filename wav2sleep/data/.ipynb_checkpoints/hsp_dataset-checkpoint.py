### this file contains functions and logic specific to HSP dataset processing 

import pandas as pd 
import numpy as np
import os 
import subprocess
from ..settings import *
from ..config import *
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
    fp_dict = {}
    if site == S0001:
        print(f'Preparing file paths for site {site}...')
        metadata_fp = os.path.join(folder, 'psg-metadata', 'I0001_psg_metadata_2025-05-06.csv')
        metadata_df = pd.read_csv(metadata_fp)
        metadata_df_filtered = metadata_df[ (metadata_df.HasSleepAnnotations == 'Y') & (metadata_df.HasStaging == 'Y') ].copy()
        output_dir = os.path.join(output_folder, dataset, site) # for master.csv, folder download and final output
        if not os.path.exists(output_dir): # make the directory if it doesn't exist already
            os.makedirs(output_dir)
        """
        Desired file structure:

        output_dir/
            - master.csv # master csv, filtered to files of interest
            - DOWNLOAD/ # location to temporarily store downloaded patient-level folders
                - subfolder/ 
                    - session/
                        - edf file, annotation files
            - INGEST/ # location to store output of preprocessing
                - subfolder/
                    - session/
                        - result of preprocessing
        """
        metadata_df_filtered.to_csv(os.path.join(output_dir, 'master.csv'), index = False, header = True) # filtered metadata file 
        for _, row in metadata_df_filtered.iterrows():
            ID = str(row.BDSPPatientID)
            subfolder = row.BidsFolder
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
    input: dataframe annotations.csv
    fs: sampling rate
    t0: start datetime from signal file, if None, from the first line of annotations
    output: dataframe annotations with new columns: event starts/ends in seconds and ends
    """
    # set missing duration
    annotations['event'] = annotations.event.astype(str)
    annotations.loc[pd.isna(annotations.duration), 'duration'] = 1/fs # set missing duration to 1/sampling rate
    
    # deal with missing time
    annotations = annotations[pd.notna(annotations.time)].reset_index(drop=True)
    
    # remove negative epochs (these usually corresponding to irrelevant labels before start of edf recording)
    if any(annotations.epoch < 1):
        annotations = annotations[annotations.epoch >= 1].reset_index(drop=True)

    if t0 is None:
        annotations['time'] = pd.to_datetime(annotations['time']).tz_localize(None)
        t0 = annotations.time.iloc[0]
    else:
        t0 = pd.to_datetime(t0).tz_localize(None)  # make sure it is 
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

def get_sleep_stage(annotations_df, annotation_indicator_dict):
    """
    Extract sleep stage labels from annotations dataframe.
    """
    # extract sleep stage labels 
    sleep_stages_events = annotations_df.loc[annotations_df.event.apply(lambda x: 'sleep_stage' in str(x).lower()), :].copy()
    if len(sleep_stages_events) == 0:
        annotation_indicator_dict['Sleep Stages'] = False
        return sleep_stages_events, annotation_indicator_dict
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
    if len(lights_off_events) != 1:
        annotation_indicator_dict[LIGHTS_OFF] = False
    else:
        if lights_on_events.start_sec.iloc[0] < lights_off_events.start_sec.iloc[0]:
            lights_on_events.loc[:, 'start_sec'] += 86400
            if lights_on_events.start_sec.iloc[0] >= annotations_df.start_sec.iloc[-1] + 2*3600:
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
        annotations_df: pd.DataFrame with columns including 'event' and 'epoch'
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
    This function assumes column names consistent with HSP S0001 site: ['epoch', 'time',  'duration', 'event']
    
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
        return None, None, None
           
    # preprocess: drop negative epochs, deal with midnight times stamps
    annotations_df = annotations_preprocess(annotations_df, fs, t0=t0)

    # annotation qc dictionary, indicating presence of each event annotation
    annotation_indicator_dict = {}
    
    # extract post-processing relevant events (1) Lights on/off and (2) 'Recording Resumed'
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

    # combine all other events into a single dataframe and sort by start_sec and reset index
    all_df = pd.concat([lights_events_df, arousal_events_df, limb_events_df, respiratory_events_df])
    all_df = all_df.sort_values(by='start_sec').reset_index(drop=True)

    # subset to event, start_sec, end_sec (we probably don't need epoch, time, duration, idx_start, idx_end anymore)
    all_df = all_df[['event', 'start_sec', 'end_sec']]

    return sleep_stage_series, body_position_series, all_df, annotation_indicator_dict

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
            return None, None, None, None, None, error

        # --- define annotation file to parse ---
        if len(annotation_files) == 1:
            annotations_df = pd.read_csv(os.path.join(download_folder, annotation_files[0]))
            # --- invoke annotation parsing function
            raw_temp = mne.io.read_raw_edf(edf_fp, preload=False) # preload = False, I just want to get the start time and sampling frequency for annotations processing
            start_time = raw_temp.info['meas_date']
            sfreq = raw_temp.info['sfreq']
            len_sec = raw_temp.n_times / sfreq
            labels, body_position_series, all_df, annotation_indicator_dict = annotations_process(annotations_df, sfreq, len_sec, t0=start_time)
            if labels is None and all_df is None:
                error = f"Failed to process annotations for {download_folder}, annotation files are missing event durations"
                return None, None, None, None, None, error
        else:
            error = f"Expected 1 annotation file, found {len(annotation_files)}"
            return None, None, None, None, None, error

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
    