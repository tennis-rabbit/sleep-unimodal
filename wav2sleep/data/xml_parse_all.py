"""Parse Sleep stages from XML files."""

import re

import numpy as np
import pandas as pd
from typing import Union, List

from wav2sleep.settings import LABEL, TIMESTAMP
from wav2sleep.data.utils import convert_int_stage, convert_int_stage_five
from ..config import EVENT_NAME_COLUMN, START_TIME_COLUMN, END_TIME_COLUMN

# used later for annotating waveforms
from wav2sleep.data.edf import INV_ALT_COLUMNS

def parse_xml_annotations(filepath) -> pd.Series:
    """Parse an annotations XML file to retrieve a series of sleep stages indexed in seconds.

    Inspired by:
    https://github.com/drasros/sleep_staging_shhs/blob/master/shhs.py
    """
    with open(filepath) as f:
        content = f.read()
    patterns_start = re.findall(r'<EventConcept>Recording Start Time</EventConcept>\n<Start>0</Start>', content)
    if len(patterns_start) == 0:
        raise ValueError(f'{filepath=} had no start time.')
    elif len(patterns_start) > 1:
        raise ValueError(f'{filepath=} had multiple start times.')
    # Find annotations within the XML via regex matching.
    stage_patterns = find_stages(content)
    return create_sleep_series(stage_patterns)


def find_stages(file_contents: str):
    """Find sleep stages within an XML using regex matching."""
    return re.findall(
        r'<EventType>Stages.Stages</EventType>\n'
        + r'<EventConcept>.+</EventConcept>\n'
        + r'<Start>.+</Start>\n'
        + r'<Duration>.+</Duration>\n'
        + r'</ScoredEvent>',
        file_contents,
    )


def create_sleep_series(stage_patterns) -> pd.Series:
    """Create pandas series of sleep stages from a list of stage patterns."""
    stages = []

    for ind, pattern in enumerate(stage_patterns):
        _, sleep_stage_str, start_str, duration_str, *_ = pattern.splitlines()
        #stage = convert_int_stage(sleep_stage_str[-16])
        stage = convert_int_stage_five(sleep_stage_str[-16])
        start = float(start_str[7:-8])
        if ind == 0 and start != 0.0:
            raise ValueError(f'First stage did not start at 0.0s: {start}')
        duration = float(duration_str[10:-11])
        if duration % 30 != 0.0:
            raise ValueError(f'Non-30s epoch duration: {duration}')
        num_epochs = int(duration) // 30
        stages += [stage] * num_epochs
    ts = np.arange(0, 30 * len(stages), 30.0)  # Timestamps in seconds from start
    # Make labels correspond to previous 30s rather than next 30s
    ts += 30
    return pd.DataFrame({LABEL: stages, TIMESTAMP: ts}).set_index(TIMESTAMP).squeeze().sort_index()


# --- Functions below are to extract other events from XML files.

# --- getting all annotations with type, concept, start, end, and signal location
def parse_all_annotations(filepath) -> pd.DataFrame:
    """Parse an annotations XML file to retrieve all events"""
    with open(filepath) as f:
        content = f.read()
    patterns_start = re.findall(r'<EventConcept>Recording Start Time</EventConcept>\n<Start>0</Start>', content)
    if len(patterns_start) == 0:
        raise ValueError(f'{filepath=} had no start time.')
    elif len(patterns_start) > 1:
        raise ValueError(f'{filepath=} had multiple start times.')
    # Find annotations within the XML via regex matching.
    all_patterns = find_all_events(content)
    # Create events dataframe
    all_df = create_event_df(all_patterns)
    return all_df

def find_all_events(file_contents: str):
    """Find all events within an XML using regex matching."""
    return re.findall(
        r'<EventType>.+</EventType>\n'
        + r'<EventConcept>.+</EventConcept>\n'
        + r'<Start>.+</Start>\n'
        + r'<Duration>.+</Duration>\n'
        + r'<SignalLocation>.+</SignalLocation>\n', 
        file_contents,
    ) # end at SignalLocation instead of ScoredEvent because some events have additional info before ScoredEvent

def create_event_df(patterns) -> pd.DataFrame:
    """Create pandas DataFrame of respiratory and arousal events from a list of patterns."""

    #event_types = []
    event_concepts = []
    starts = []
    ends = []
    #signal_locs = []

    for ind, pattern in enumerate(patterns):
        event_type_str, event_concept_str, start_str, duration_str, signal_loc_str, *_ = pattern.splitlines()
        #event_type = re.split(r'[^a-zA-Z0-9 ]+', event_type_str[11:-12])[0] # splits at any non-numeric, non-alphabet character (excluding spaces)
        # event_concept = re.split(r'[^a-zA-Z0-9 ]+', event_concept_str[14:-15])[0]
        event_concept = event_concept_str[14:-15].strip()
        start = float(start_str[7:-8])
        duration = float(duration_str[10:-11])
        end = start + duration
        #signal_loc = signal_loc_str[16:-17]

        #event_types += [event_type]
        event_concepts += [event_concept]
        starts += [start]
        ends += [end]
        #signal_locs += [signal_loc]
        
    #df = pd.DataFrame({
    #    "Types": event_types, 
    #    "Concepts": event_concepts,
    #    "Starts": starts,
    #    "Ends": ends,
    #    "Signal": signal_locs
    #})

    df = pd.DataFrame({
        EVENT_NAME_COLUMN: event_concepts,
        START_TIME_COLUMN: starts,
        END_TIME_COLUMN: ends,
    })

    return df
# ---

# --- merge annotations to signal data
def annotate_waveform(
    waveform_df,
    annotations_dfs
) -> pd.DataFrame:
    """
    Annotate waveform DataFrame with events from one or more annotation DataFrames.

    Args:
        waveform_df: The waveform data with time as index.
        annotations_dfs: One or more annotation DataFrames. Each must contain
                         'Concepts', 'Signal', 'Starts', and 'Ends' columns.

    Returns:
        Annotated waveform DataFrame with new columns for each signal.
    """
    waveform_df_new = waveform_df.copy()

    # Normalize to list
    if isinstance(annotations_dfs, pd.DataFrame):
        annotations_dfs = [annotations_dfs]

    # Concatenate all annotation dataframes (ensure they have the right columns)
    required_cols = ['Concepts', 'Signal', 'Starts', 'Ends']
    all_annotations = pd.concat(
        [df[required_cols] for df in annotations_dfs],
        ignore_index=True
    )

    # Go through each annotation
    for _, row in all_annotations.iterrows():

        concept = row['Concepts']
        signal = row['Signal']
        start = row['Starts']
        end = row['Ends']

        event_col_name = INV_ALT_COLUMNS[signal] + "_events"

        # Initialize column if it doesn't exist
        if event_col_name not in waveform_df_new.columns:
            waveform_df_new[event_col_name] = pd.Series(np.nan, index=waveform_df_new.index, dtype='object')

        # Get mask of rows within the start–end interval
        mask = (waveform_df_new.index >= start) & (waveform_df_new.index <= end)

        # Fill in the annotation: merge if multiple concepts exist
        def merge_concepts(existing):
            if pd.isna(existing):
                return concept
            if concept in existing.split(','):
                return existing  # already there
            return f"{existing},{concept}" 
        
        waveform_df_new.loc[mask, event_col_name] = waveform_df_new.loc[mask, event_col_name].apply(merge_concepts)
        #print("start replacing")
    waveform_df_new.replace({None: np.nan}, inplace=True)
    # print(waveform_df_new)
    return waveform_df_new
# ---


# --- parsing unique events to check what kind of annotations are available (not relevant to preprocessing pipeline. only for exploration)
def parse_unique_events_concepts(filepath):
    with open(filepath) as f:
        content = f.read()
    patterns_start = re.findall(r'<EventConcept>Recording Start Time</EventConcept>\n<Start>0</Start>', content)
    if len(patterns_start) == 0:
        raise ValueError(f'{filepath=} had no start time.')
    elif len(patterns_start) > 1:
        raise ValueError(f'{filepath=} had multiple start times.')
    # Find annotations within the XML via regex matching.
    any_patterns = find_any_events(content)
    # Create events dataframe
    types, concepts = get_unique_events(any_patterns)
    return types, concepts

def find_any_events(file_contents: str):
    """Find any event in XML with an EventType and EventConcept."""
    return re.findall(
        r'<EventType>.+</EventType>\n'
        + r'<EventConcept>.+</EventConcept>\n', 
        file_contents,
    ) 

def get_unique_events(patterns):
    """Get a list of unique event types and event concepts in patterns"""

    event_types = []
    event_concepts = []

    for ind, pattern in enumerate(patterns):
        event_type_str, event_concept_str, *_ = pattern.splitlines()
        event_type = re.split(r'[^a-zA-Z0-9 ]+', event_type_str[11:-12])[0] # splits at any non-numeric, non-alphabet character (excluding spaces)
        event_concept = re.split(r'[^a-zA-Z0-9 ]+', event_concept_str[14:-15])[0]

        event_types += [event_type]
        event_concepts += [event_concept]
        
    unique_types = list(set(event_types))
    unique_concepts = list(set(event_concepts))

    return unique_types, unique_concepts
# ---
