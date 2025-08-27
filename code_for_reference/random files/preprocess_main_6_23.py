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

from wav2sleep.data.edf import load_edf_data
from wav2sleep.data.txt import parse_txt_annotations
from wav2sleep.data.utils import interpolate_index
from wav2sleep.data.xml import parse_xml_annotations
from wav2sleep.parallel import parallelise


import pandas as pd
import numpy as np
import os
from wav2sleep.data.edf import load_edf_data
from wav2sleep.data.txt import parse_txt_annotations
from wav2sleep.data.utils import interpolate_index
from wav2sleep.data.xml import parse_xml_annotations
from wav2sleep.settings import *
from wav2sleep.config import *


logger = logging.getLogger(__name__)




def _mne_highpass_series(s: pd.Series, fs,
                         cutoff = None) -> pd.Series:
   
    if cutoff is None:
        return s                       

    
    x  = s.to_numpy(np.float64)[np.newaxis, :]             # shape (1, n)

    x_filt = mne.filter.filter_data(
        x, sfreq=fs,
        l_freq=cutoff, h_freq=None,
        method='fir', phase='zero-double',
        n_jobs='cuda',  # GPU or CPU
        verbose=False
    )[0]

    return pd.Series(x_filt, index=s.index, name=s.name)
def process_edf(edf: pd.DataFrame):
    """Process dataframe of EDF data."""
    signals = []

    def _process_edf_column(col, target_index, preprocessed_fs):
        """Process signal column of EDF"""
        if col in edf:
            
            raw = edf[col].dropna()
            
            # print(len(raw.loc[0:1]))
            raw_fs = len(raw.loc[0:1]) - 1
            
            if raw_fs >= preprocessed_fs: 
                raw_hp = _mne_highpass_series(raw, raw_fs, cutoff = preprocessed_fs/2)
            else:
                raw_hp = raw
            
            resampled = interpolate_index(raw_hp, target_index,
                              method="linear", squeeze=False)
            # normalized_wav = (resampled_wav - resampled_wav.mean()) / resampled_wav.std()
            print("col:", col, "length:", resampled.shape)
            signals.append(resampled)
            return 0
        else:
            return 1

    _process_edf_column(ECG, ECG_SIGNAL_INDEX, FREQ_ECG)
    _process_edf_column(HR, HR_SIGNAL_INDEX, FREQ_HR)

    _process_edf_column(SPO2, SPO2_SIGNAL_INDEX, FREQ_SPO2)
    _process_edf_column(OX, OX_SIGNAL_INDEX, FREQ_OX)
    _process_edf_column(ABD, ABD_SIGNAL_INDEX, FREQ_ABD)
    _process_edf_column(THX, THX_SIGNAL_INDEX, FREQ_THX)
    _process_edf_column(AF, AF_SIGNAL_INDEX, FREQ_AF)
    _process_edf_column(NP, NP_SIGNAL_INDEX, FREQ_NP)
    _process_edf_column(SN, SN_SIGNAL_INDEX, FREQ_SN)
    
    _process_edf_column(EMG_LLeg, EMG_LLeg_SIGNAL_INDEX, FREQ_EMG_LLeg)
    _process_edf_column(EMG_RLeg, EMG_RLeg_SIGNAL_INDEX, FREQ_EMG_RLeg)
    _process_edf_column(EMG_LChin, EMG_LChin_SIGNAL_INDEX, FREQ_EMG_LChin)
    _process_edf_column(EMG_RChin, EMG_RChin_SIGNAL_INDEX, FREQ_EMG_RChin)
    _process_edf_column(EMG_CChin, EMG_CChin_SIGNAL_INDEX, FREQ_EMG_CChin)
    _process_edf_column(EOG_L, EOG_L_SIGNAL_INDEX, FREQ_EOG_L)
    _process_edf_column(EOG_R, EOG_R_SIGNAL_INDEX, FREQ_EOG_R)
    
    is_na_C3 = _process_edf_column(EEG_C3, EEG_C3_SIGNAL_INDEX, FREQ_EEG_C3)
    is_na_C4 = _process_edf_column(EEG_C4, EEG_C4_SIGNAL_INDEX, FREQ_EEG_C4)
    is_na_A1 = _process_edf_column(EEG_A1, EEG_A1_SIGNAL_INDEX, FREQ_EEG_A1)
    is_na_A2 = _process_edf_column(EEG_A2, EEG_A2_SIGNAL_INDEX, FREQ_EEG_A2)
    is_na_O1 = _process_edf_column(EEG_O1, EEG_O1_SIGNAL_INDEX, FREQ_EEG_O1)
    is_na_O2 = _process_edf_column(EEG_O2, EEG_O2_SIGNAL_INDEX, FREQ_EEG_O2)
    is_na_F3 = _process_edf_column(EEG_F3, EEG_F3_SIGNAL_INDEX, FREQ_EEG_F3)
    is_na_F4 = _process_edf_column(EEG_F4, EEG_F4_SIGNAL_INDEX, FREQ_EEG_F4)
    
    # add a logic to check
    
    is_na_C3_A2 = _process_edf_column(EEG_C3_A2, EEG_C3_A2_SIGNAL_INDEX, FREQ_EEG_C3_A2)
    is_na_C4_A1 = _process_edf_column(EEG_C4_A1, EEG_C4_A1_SIGNAL_INDEX, FREQ_EEG_C4_A1)
    is_na_F3_A2 = _process_edf_column(EEG_F3_A2, EEG_F3_A2_SIGNAL_INDEX, FREQ_EEG_F3_A2)
    is_na_F4_A1 = _process_edf_column(EEG_F4_A1, EEG_F4_A1_SIGNAL_INDEX, FREQ_EEG_F4_A1)
    is_na_O1_A2 = _process_edf_column(EEG_O1_A2, EEG_O1_A2_SIGNAL_INDEX, FREQ_EEG_O1_A2)
    is_na_O2_A1 = _process_edf_column(EEG_O2_A1, EEG_O2_A1_SIGNAL_INDEX, FREQ_EEG_O2_A1)
    
    
    
    merged_df = pd.concat(signals, axis=1).astype(np.float32)
    
    if (EEG_C3_A2 not in merged_df.columns.to_list()) and (is_na_C3 == 0) and (is_na_A2 == 0):
        merged_df[EEG_C3_A2] = merged_df[EEG_C3] - merged_df[EEG_A2]
    if (EEG_C4_A1 not in merged_df.columns.to_list()) and (is_na_C4 == 0) and (is_na_A1 == 0):
        merged_df[EEG_C4_A1] = merged_df[EEG_C4] - merged_df[EEG_A1]
    if (EEG_F3_A2 not in merged_df.columns.to_list()) and (is_na_F3 == 0) and (is_na_A2 == 0):
        merged_df[EEG_F3_A2] = merged_df[EEG_F3] - merged_df[EEG_A2]
    if (EEG_F4_A1 not in merged_df.columns.to_list()) and (is_na_F4 == 0) and (is_na_A1 == 0):
        merged_df[EEG_F4_A1] = merged_df[EEG_F4] - merged_df[EEG_A1]
    if (EEG_O1_A2 not in merged_df.columns.to_list()) and (is_na_O1 == 0) and (is_na_A2 == 0):
        merged_df[EEG_O1_A2] = merged_df[EEG_O1] - merged_df[EEG_A2]
    if (EEG_O2_A1 not in merged_df.columns.to_list()) and (is_na_O2 == 0) and (is_na_A1 == 0):
        merged_df[EEG_O2_A1] = merged_df[EEG_O2] - merged_df[EEG_A1]    
    
    merged_df = (merged_df - merged_df.mean()) / merged_df.std()
    return merged_df


def process(edf_fp: str, label_fp: str, output_fp: str, overwrite: bool = False) -> bool:
    """Process night of data."""
    if os.path.exists(output_fp) and not overwrite:
        logger.debug(f'Skipping {edf_fp=}, {output_fp=}, already exists')
        return False
    else:
        os.makedirs(os.path.dirname(output_fp), exist_ok=True)

    # Process labels
    if label_fp.endswith('.xml'):
        try:
            labels = parse_xml_annotations(label_fp)
        except Exception as e:
            logger.error(f'Failed to parse: {label_fp}.')
            logger.error(e)
            return False
    else:
        labels = parse_txt_annotations(fp=label_fp)
        if labels is None:
            logger.error(f'Failed to parse: {label_fp}.')
            return False
    labels = labels.reindex(TARGET_LABEL_INDEX).fillna(-1)
    # Check for N1, N3 or REM presence. (Recordings with just sleep-wake typically use N2 as sole sleep class)
    stage_counts = labels.value_counts()
    
    if stage_counts.get(1.0) is None and stage_counts.get(3.0) is None and stage_counts.get(4.0) is None:
        
        logger.error(f'No N1, N3 or REM in {label_fp}.')
        output_fp = output_fp.replace('.parquet', '.issues.parquet')
    edf = load_edf_data(edf_fp, columns=EDF_COLS, raise_on_missing=False)
    waveform_df = process_edf(edf)
    output_df = pd.concat([waveform_df, labels], axis=1)
    
    output_df.to_parquet(output_fp)
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
        for edf_fp in edf_fps:
            all_score_fp = edf_fp.replace('.edf', '.allscore.txt')
            stg_fp = edf_fp.replace('.edf', '.stg.txt')
            if os.path.exists(stg_fp):
                label_fp = stg_fp
            elif os.path.exists(all_score_fp):
                label_fp = all_score_fp
            else:
                continue
            session_id = os.path.basename(edf_fp).replace('.edf', '')
            output_fp = os.path.join(output_folder, dataset, INGEST, f'{session_id}.parquet')
            fp_dict[session_id] = {'edf_fp': edf_fp, 'label_fp': label_fp, 'output_fp': output_fp}
        return fp_dict
    
    
    # Other datasets use NSRR standardized XML files
    label_fps = glob(f'{folder}/polysomnography/annotations-events-nsrr/**/**.xml', recursive=True)
    for label_fp in label_fps:
        session_id = os.path.basename(label_fp).replace('-nsrr.xml', '')
        edf_fp = get_edf_path(session_id, dataset, folder)
        if not os.path.exists(edf_fp):
            logger.warning(f"{edf_fp=} doesn't exist. Skipping...")
            continue
        output_fp = os.path.join(output_folder, dataset, INGEST, f'{session_id}.parquet')
        fp_dict[session_id] = {'edf_fp': edf_fp, 'label_fp': label_fp, 'output_fp': output_fp}
    return fp_dict
############### this two functions are related to the file structure!!!!!!!!! check this please!!!!!!!!!!!!!!

def process_files(
    fp_dict, max_parallel: int = 1, overwrite: bool = False, address: str = 'local'
):
    print(f'Preparing to process {len(fp_dict)} files.')

    def proc(arg_dict):
        try:
            return process(overwrite=overwrite, **arg_dict)
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
            num_converted += process(**fp_map)
    print(f'Converted {num_converted} files.')


def parse_args():
    parser = argparse.ArgumentParser(prog='Dataset Processor', description='Process dataset.')
    parser.add_argument('--folder', help='Location of dataset.')
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
    fp_dict = prepare_dataset(folder=args.folder, output_folder=args.output_folder, dataset=dataset)
    process_files(fp_dict, max_parallel=args.max_parallel, overwrite=args.overwrite, address=args.cluster_address)


if __name__ == '__main__':
    main()
