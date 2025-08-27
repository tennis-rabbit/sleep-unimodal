import pandas as pd
import numpy as np


def parse_process_rpoints_annotations(rpoints_fp: str) -> pd.DataFrame:
    """Parse and process rpoints annotations files, which are annotations of heartbeats in ECG data
    SHHS ONLY for now"""

    # read csv file 
    rpoints = pd.read_csv(rpoints_fp)
    rpoints.rename(columns={'seconds': 'RPoint_time'}, inplace=True)

    # calculate R points and T points times
        # note that all files, regardless of actual sampling frequency, are annotated by sample number as if the sampling frequency was 256
        # very weird, but this is what it says in SHHS documentation, and it is consistent with the data
    # if no P or T wave detected, PPoint and TPoint are -1
        # let's say for now, we only want to keep observed QRS complex points
        # there is a possibility that we want to indicate missingness of P and T points, but we should differentiate it from non-annotated (most P waves are there but not annotated)
    rpoints.loc[:,'PPoint'] = rpoints['PPoint'].replace(-1, np.nan) # if missing, set to np.nan
    rpoints.loc[:,'TPoint'] = rpoints['TPoint'].replace(-1, np.nan) # if missing, set to np.nan
    rpoints.loc[:,'PPoint_time'] = rpoints['PPoint']/256
    rpoints.loc[:,'TPoint_time'] = rpoints['TPoint']/256

    # turn types into corresponding strings
    type_map = {
        1: 'Normal Sinus Beat',
        2: 'Ventricular Ectopic Beat (VE)',
        3: 'Supraventricular Ectopic Beat (SVE)',
        0: 'Artifact'
    }
    rpoints.loc[:,'BeatType'] = rpoints['Type'].map(type_map)
    
    # other potentially important columns? 
        # 'STLevel1', 'STSlope1', 'STLevel2', 'STSlope2', 'Manual'
    r_annots = rpoints[['RPoint_time', 'BeatType']].dropna().rename(columns={'RPoint_time': 'time'})
    r_annots['annot'] = 'R-' + r_annots['BeatType']
    p_annots = rpoints[['PPoint_time', 'BeatType']].dropna().rename(columns={'PPoint_time': 'time'})
    p_annots['annot'] = 'P-' + p_annots['BeatType']
    t_annots = rpoints[['TPoint_time', 'BeatType']].dropna().rename(columns={'TPoint_time': 'time'})
    t_annots['annot'] = 'T-' + t_annots['BeatType']
    annots_df = pd.concat([r_annots[['time', 'annot']],
                      p_annots[['time', 'annot']],
                      t_annots[['time', 'annot']]])
    annots_df = annots_df.sort_values(by='time').reset_index(drop=True)

    return annots_df
