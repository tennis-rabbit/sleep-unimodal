import os
from glob import glob

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import mne
from ..config import *
from scipy.signal import convolve

def get_parquet_cols(fp: str):
    """Get list of column names from parquet file."""
    cols = pq.read_schema(fp, memory_map=True).names
    if '__index_level_0__' in cols:
        cols.remove('__index_level_0__')
    return cols


def get_parquet_fps(folder: str):
    """Return parquet files in a folder."""
    if not os.path.exists(folder):
        raise FileNotFoundError(folder)
    return glob(f'{folder}/*.parquet')


def convert_int_stage(stage):
    stage = int(stage)
    if stage not in [0, 1, 2, 3, 4, 5, 6, 7, 9]:  # 6 = mvmt, 9 = unscored
        raise ValueError(f'{stage} not a valid sleep stage.')
    # Map any N4 to 3 (N3), REM to 4.
    if stage == 4:
        stage = 3
    elif stage == 5:
        stage = 4
    elif stage in [6, 7, 9]:
        stage = np.nan
    return stage


def convert_str_stage(stage: str):
    if 'STAGE' not in stage:
        return np.nan
    if 'NO STAGE' in stage:
        return np.nan
    elif 'W' in stage:
        return 0
    elif 'N1' in stage:
        return 1
    elif 'N2' in stage:
        return 2
    elif 'N3' in stage:
        return 3
    elif 'R' in stage:
        return 4
    elif 'MVT' in stage:
        return np.nan
    else:
        raise ValueError(f'Encountered unseen value: {stage=}')

# --- new sleep stage conversions ---
SLEEP_STAGE_INVERSE_MAPPING = {v: k for k, values in EVENT_NAME_MAPPING['Sleep Stages'].items() for v in values}

def convert_int_stage_five(stage):
    """
    Convert sleep stage integer to five-level sleep stage: unscores, wake, light, deep, rem (see config.py)
    """
    stage = int(stage)
    if stage not in [0, 1, 2, 3, 4, 5, 6, 7, 9]:  # 6 = mvmt, 9 = unscored
        raise ValueError(f'{stage} not a valid sleep stage.')
    return SLEEP_STAGE_INVERSE_MAPPING[stage]

def convert_str_stage_five(stage: str):
    """
    Convert sleep stage string to five-level sleep stage: unscores, wake, light, deep, rem (see config.py)
    """
    if 'STAGE' not in stage:
        return SLEEP_STAGE_UNKNOWN
    if 'NO STAGE' in stage:
        return SLEEP_STAGE_UNKNOWN
    elif 'W' in stage:
        return SLEEP_STAGE_WAKE
    elif 'N1' in stage:
        return SLEEP_STAGE_LIGHT_SLEEP
    elif 'N2' in stage:
        return SLEEP_STAGE_LIGHT_SLEEP
    elif 'N3' in stage:
        return SLEEP_STAGE_DEEP_SLEEP
    elif 'R' in stage:
        return SLEEP_STAGE_REM_SLEEP
    elif 'MVT' in stage:
        return SLEEP_STAGE_UNKNOWN
    else:
        raise ValueError(f'Encountered unseen value: {stage=}')

# --- 


def interpolate_index(
    source_df,
    target_index,
    method,
    squeeze,
    **kwargs,
):
    """Re-sample pandas Data (Series or DataFrame) to match a target index.

    This function takes an input pandas series or dataframe (source_df)
    and interpolates/resamples it to align with a target_index

    kwargs are passed as parameters to the interpolate method:
    https://pandas.pydata.org/docs/reference/api/pandas.Series.interpolate.html
    """
    print(target_index.__class__, source_df.index.__class__)
    if target_index.__class__ != source_df.index.__class__:
        raise ValueError('target_index must be the same type as the source_index.')
    if method is None:
        if isinstance(source_df.index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
            method = 'time'
            # Check for timezone info - think this still needs to be removed before for interpolation.
            if hasattr(source_df.index, 'tz_localize'):
                raise ValueError('sourcet_index contains TZ information. Remove and reinstate for interpolation.')
            elif hasattr(target_index, 'tz_localize'):
                raise ValueError('target_index contains TZ information. Remove and reinstate for interpolation.')
        else:
            method = 'index'
    # Add NaNs at interpolation timestamps where no data is available
    if isinstance(source_df, pd.Series):
        source_df = source_df.to_frame()
    nan_padded_df = source_df.join(pd.DataFrame(index=target_index), how='outer')
    # Re-sample dataframe at timestamps then slice at the interpolated values
    resampled_df = nan_padded_df.interpolate(method=method, limit_direction='both', **kwargs).loc[target_index]
    if squeeze:
        return resampled_df.squeeze(axis='columns')
    else:
        return resampled_df


def mne_lowpass_series(s: pd.Series, fs,
                        cutoff=None, highpass_cutoff = None) -> pd.Series:
    """
    Apply low-pass filter to a pd.Series using MNE.
    Keeps frequencies below the cutoff.
    
    Parameters:
    - s: input signal
    - fs: sampling rate
    - cutoff: cutoff frequency (Hz)
    """
    if (cutoff is None) and (highpass_cutoff is None):
        return s

    x = s.to_numpy(np.float64)[np.newaxis, :]  # shape (1, n)

    x_filt = mne.filter.filter_data(
        x, sfreq=fs,
        l_freq=highpass_cutoff, h_freq=cutoff,  
        method='fir', phase='zero-double',
        n_jobs='cuda',
        verbose=False
    )[0]

    return pd.Series(x_filt, index=s.index, name=s.name)


def local_normalization(input_data, length=100): # from Yuzhe's code
    """
    Local normalization of a signal.
    Args:
        length: length of the kernel
        input_data: input data
    Returns:
        normalized data
    """
    assert length % 2 == 0
    ave_kernel = np.ones((length,), dtype='float32') / length
    local_mean = convolve(input_data, ave_kernel, mode='same')
    residual = input_data - local_mean
    residual_square = residual ** 2 # residual square
    local_std = convolve(residual_square, ave_kernel, mode='same') ** 0.5 + 1e-30 # variance to standard deviation 
    return np.divide(residual, local_std) # normalize by standard deviation