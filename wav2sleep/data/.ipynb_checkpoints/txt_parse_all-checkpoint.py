"""Code to convert Wisconsin Sleep Cohort (WSC) sleep stage annotations to consistent format."""

import datetime
import logging
import os

import numpy as np
import pandas as pd

from wav2sleep.settings import LABEL, TIMESTAMP
from wav2sleep.data.edf import get_edf_end, get_edf_start
from wav2sleep.data.utils import convert_int_stage, convert_str_stage, convert_str_stage_five, convert_int_stage_five
from ..config import *
logger = logging.getLogger(__name__)

# Check recordings between 4 and 14 hours long.
MIN_RECORDING_LENGTH = 60 * 60 * 4
MAX_RECORDING_LENGTH = 60 * 60 * 14


def convert_index(hour_minute_index, start_ts: datetime.datetime) -> pd.DatetimeIndex:
    """Convert hh:mm:.. string timestamp index to datetimes.

    This function expects recordings to start between 5pm and 3am to correctly convert
    timestamps in hh:mm format to the correct datetime.
    """
    # Get hour for each timestamp.
    hours = hour_minute_index.str.slice(start=0, stop=2).astype(int)
    # Assuming recording is less than 24h, should span two days if end hour is less than start.
    end_hour = hours[-1]
    if start_ts.hour < end_hour:
        multiday = False
    else:
        multiday = True
    start_day = start_ts.date()
    start_day_str = start_day.strftime('%Y/%m/%d')
    end_day = start_day + datetime.timedelta(days=1)
    end_day_str = end_day.strftime('%Y/%m/%d')
    same_day = start_ts.hour <= hours
    if not multiday and ((~same_day).any()):
        raise ValueError(f'{multiday=}, but {start_ts=} and found timestamps on a possibly different day.')
    day_strings = np.repeat(start_day_str, len(hour_minute_index))
    day_strings[~same_day] = end_day_str
    string_index = day_strings + (' ' + hour_minute_index)
    datetime_index = pd.DatetimeIndex(string_index)
    return datetime_index


################################################################

import re
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

TIME_FMTS = ("%H:%M:%S.%f", "%H:%M:%S")        
EPOCH_LEN  = 30                                 

def _parse_time(s: str) -> datetime.time:
    for fmt in TIME_FMTS:
        try:
            return datetime.strptime(s, fmt).time()
        except ValueError:
            continue
    raise ValueError(f"Un-parsable time string: {s!r}")

def _safe_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except ValueError:
        return default

def parse_sco_wsc(fp) -> pd.DataFrame:
    
    events = []
    fp = Path(fp)
    with fp.open() as fh:
        for ln in fh:
            ln = ln.rstrip()
            
            if not ln or ln.startswith(("Epoch", "*", "#")):
                continue

            
            cols = re.split(r"\t", ln) # should use "\t" instead of "\t+" pr "\s"
            if len(cols) < 6:          
                continue
            
            if cols[0] == '':
                continue
            epoch        = int(cols[0])
            marker_code  = cols[3]     # e.g. 405
            marker_text  = cols[4].strip()   # e.g. LMA
            clock_str    = cols[6]     # e.g. 21:29:42 or 21:29:42.123
            
            duration_sec = 0.0
            
            
            duration_sec = float(cols[-1])
            if duration_sec > 0:
                
                events.append((epoch, marker_text, clock_str, duration_sec))

    if not events:
        raise ValueError(f"No valid marker rows found in {fp}")
    
    
    first_epoch, _, first_clock, _ = events[0]
    first_abs_dt = datetime.combine(
        datetime(2000, 1, 1), _parse_time(first_clock)
    )
    study_start_abs = first_abs_dt - timedelta(seconds=first_epoch * EPOCH_LEN)
    
    
    rows = []
    for epoch, ev, t_str, dur in events:
        abs_dt = datetime.combine(study_start_abs.date(), _parse_time(t_str))
        start_sec = (abs_dt - study_start_abs).total_seconds()
        end_sec   = start_sec + dur
        if start_sec<0:
            start_sec += 86400
        if start_sec>=86400:
            start_sec -= 86400
        if end_sec<0:
            end_sec += 86400
        if end_sec>=86400:
            end_sec -= 86400 
        rows.append((ev, start_sec, end_sec, dur, epoch))

    df = pd.DataFrame(
        rows,
        columns=[EVENT_NAME_COLUMN, START_TIME_COLUMN, END_TIME_COLUMN, "duration_sec", "epoch"], # can be removed, the last two columns
    )
    
    df = df[[EVENT_NAME_COLUMN, START_TIME_COLUMN, END_TIME_COLUMN]]
    return df


def parse_all_score_wsc(fp) -> pd.DataFrame:
    
    fp = Path(fp)
    start_dt: datetime | None = None
    day_offset = 0

    events = []   # (event, concept, start_sec, end_sec, dur, epoch)


    rx = re.compile(
        r"""^(?P<event>.+?)\s*-\s*DUR:\s*(?P<dur>\d+(?:\.\d+)?)\s*SEC\.?\s*-\s*
            (?P<concept>[^-]+?)\s*(?:-|$)""",
        re.I | re.VERBOSE,
    )

    with fp.open() as fh:
        for ln in fh:
            ln = ln.rstrip("\n")
            if not ln or ln.startswith(("*", "#")):
                continue

            try:
                clock_str, note = ln.split("\t", 1)
            except ValueError:
                continue
            clock_str, note = clock_str.strip(), note.strip()


            if "START RECORDING" in note.upper():
                if start_dt is not None:
                    raise ValueError(f"Multiple START RECORDING lines in {fp}")
                start_dt = datetime.combine(
                    datetime(2000, 1, 1), _parse_time(clock_str)
                )
                continue

            if start_dt is None:
                continue

            m = rx.search(note)
            if not m:
                continue

            event   = m.group("event").strip()
            dur     = float(m.group("dur"))
            concept = m.group("concept").strip()

            evt_time = _parse_time(clock_str)
            evt_dt   = datetime.combine(start_dt.date() + timedelta(days=day_offset),
                                        evt_time)
            if evt_dt < start_dt:          
                day_offset += 1
                evt_dt += timedelta(days=1)

            start_sec = (evt_dt - start_dt).total_seconds()
            end_sec   = start_sec + dur
            epoch     = int(start_sec // EPOCH_LEN)

            events.append((event, concept, start_sec, end_sec, dur, epoch))

    if start_dt is None:
        raise ValueError(f"No START RECORDING line found in {fp}")
    if not events:
        raise ValueError(f"No valid DUR events found in {fp}")

    df = pd.DataFrame(
        events,
        columns=[
            "EVENT_CLASS", EVENT_NAME_COLUMN,
            START_TIME_COLUMN, END_TIME_COLUMN, "DURATION_SEC", "EPOCH",
        ],
    )
    df.loc[df["EVENT_CLASS"] == "DESATURATION", EVENT_NAME_COLUMN] = "DESATURATION"
    df.loc[df["EVENT_CLASS"] == "LM", EVENT_NAME_COLUMN] = "LM-" + df.loc[df["EVENT_CLASS"] == "LM", EVENT_NAME_COLUMN]
    df.loc[df["EVENT_CLASS"] == "AROUSAL", EVENT_NAME_COLUMN] = "AROUSAL-" + df.loc[df["EVENT_CLASS"] == "AROUSAL", EVENT_NAME_COLUMN]
    
    df = df[[EVENT_NAME_COLUMN, START_TIME_COLUMN, END_TIME_COLUMN]]
    
    return df

####################################################################


def midnight_dist(hh_mm_string: str):
    hour = float(hh_mm_string[0][:2])
    minute = float(hh_mm_string[0][3:5])
    return (hour + minute / 60 - 24) % 24


def get_start_from_log(fp: str) -> pd.Series:
    """Determine the start of the recording from the .log.txt file.

    For some recordings, there are restarts, so we assume the closest to
    midnight is the true start then check that matches the EDF.

    This process could be cleaner, and does discard a small number of EDF files.
    """
    with open(fp, 'r') as f:
        log_contents = f.readlines()
    starts = []
    for line in log_contents:
        contents = line.strip().split('\t')
        if len(contents) < 2:
            continue
        time_epoch, annotation, *_ = contents
        if annotation == 'Recording Started':
            time, epoch_no = time_epoch.split(' ', maxsplit=1)
            starts.append((time.strip(), epoch_no.strip()))
    if not bool(starts):
        logger.warning(f"Couldn't find 'Recording Start' annotation in {fp=}")
        return None, None
    if len(starts) > 1:
        logger.warning(f'Found multiple starts in {fp=}. Using closest to midnight...')
        starts = sorted(starts, key=midnight_dist)
    return starts[-1]


COL = 'User-Defined Stage'


def parse_stg_file_wsc(fp, convert_time: bool = False):
    # "wsc-visit1-10119-nsrr.stg.txt"
    df = pd.read_csv(fp, index_col=0, delimiter='\t').squeeze()
    # Some files are seemingly missing the header.
    if COL in df.columns:
        df = df[COL]
    else:
        df = pd.read_csv(fp, index_col=0, delimiter='\t', names=[COL, 'X'])[COL]
    df = df.rename(LABEL)
    # Get start time from log.
    log_fp = fp.replace('stg', 'log')
    if not os.path.exists(log_fp):
        raise FileNotFoundError(f"Couldn't find corresponding log file for {fp=}")
    start_time, epoch = get_start_from_log(log_fp)
    if start_time is None:
        return None
    hour, minute, second = map(int, start_time.split(':'))
    # Check EDF start time.
    edf_fp = fp.replace('stg.txt', 'edf')
    edf_start, edf_end = get_edf_start(edf_fp), get_edf_end(edf_fp)
    if edf_start.hour != hour or edf_start.minute != minute or edf_start.second != second:
        logger.warning(f'{edf_start=} did not match log file start: {start_time} for {fp=}. Skipping...')
        return None
    edf_duration = (edf_end - edf_start).total_seconds()
    if edf_duration < MIN_RECORDING_LENGTH:
        logger.warning(f'EDF less than {MIN_RECORDING_LENGTH=} for {fp=}')
        return None
    elif edf_duration > MAX_RECORDING_LENGTH:
        logger.warning(f'EDF greater than {MAX_RECORDING_LENGTH=} for {fp=}')
        return None
    # Convert epoch index to seconds from start.
    # Start is 1st epoch so labels already correspond to right bin edge.
    df.index *= 30.0
    #df = df.map(convert_int_stage)
    df = df.map(convert_int_stage_five)
    stage_counts = df.value_counts(dropna=False)
    # Check for N1, N3 or REM presence. (Recordings with just sleep-wake typically use N2 as sole sleep class)
    if stage_counts.get(1.0) is None and stage_counts.get(3.0) is None and stage_counts.get(4.0) is None:
        print(df)
        print(stage_counts)
        print(fp)
        raise ValueError
    if convert_time:
        df.index = edf_start + pd.TimedeltaIndex(df.index, unit='s')
    return df


def parse_txt_annotations_wsc(fp: str):
    """Parse annotations from a WSC .txt file"""
    if fp.endswith('.stg.txt'):
        return parse_stg_file_wsc(fp)
    elif fp.endswith('.sco.txt'):
        return parse_all_score_wsc(fp)
    else:
        
        raise ValueError(f'File extension {fp=} unsupported. Expected .stg.txt or .sco.txt')