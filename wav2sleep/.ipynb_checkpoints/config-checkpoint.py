import pandas as pd
import numpy as np
##### PSG datasets
# define for file structure

# those have xml nsrr annotations
SHHS = 'shhs'
CHAT = 'chat'
MROS = 'mros'
CCSHS = 'ccshs'
CFS = 'cfs'
MESA = 'mesa'
SOF = 'sof'

# ????? ...... fine
WSC = 'wsc'
HSP = 'hsp'
NCHSDB = 'nchsdb'

STAGES = 'stages'
PATS = 'pats'


##### HSP sites 
S0001 = 'S0001'
I0002 = 'I0002'
I0003 = 'I0003'
I0004 = 'I0004'
I0006 = 'I0006'
VALID_SITES = [S0001, I0002, I0003, I0004, I0006]

##### data path
META_PATH = '/scratch/besp/shared_data'

MASTER_SHHS = [
    META_PATH + "/" + SHHS + "/datasets/shhs-harmonized-dataset-0.21.0.csv",
]
MASTER_CHAT = [
    META_PATH + "/" + CHAT + "/datasets/chat-harmonized-dataset-0.14.0.csv",
]
MASTER_MROS = [
    META_PATH + "/" + MROS + "/datasets/mros-visit1-harmonized-0.6.0.csv",
]
MASTER_CCSHS = [
    META_PATH + "/" + CCSHS + "/datasets/ccshs-trec-harmonized-0.8.0.csv",
]
MASTER_CFS = [
    META_PATH + "/" + CFS + "/datasets/cfs-visit5-harmonized-dataset-0.7.0.csv",
]
MASTER_MESA = [
    META_PATH + "/" + MESA + "/datasets/mesa-sleep-harmonized-dataset-0.7.0.csv",
]
MASTER_SOF = [
    META_PATH + "/" + SOF + "/datasets/sof-visit-8-harmonized-dataset-0.8.0.csv",
]
MASTER_HSP = [ # depends on site, also I will rewrite to filtered set of nights during the ingestion script
    META_PATH + "/" + HSP + "/psg-metadata/I0001_psg_metadata_2025-05-06.csv",
    META_PATH + "/" + HSP + "/psg-metadata/I0002_psg_metadata_2025-05-06.csv",
    META_PATH + "/" + HSP + "/psg-metadata/I0003_psg_metadata_2025-05-06.csv", 
    META_PATH + "/" + HSP + "/psg-metadata/I0004_psg_metadata_2025-05-06.csv",
    META_PATH + "/" + HSP + "/psg-metadata/I0006_psg_metadata_2025-05-06.csv",
]
MASTER_CSV_LIST = {'shhs': MASTER_SHHS, 'chat': MASTER_CHAT, 'mros': MASTER_MROS, 'ccshs': MASTER_CCSHS, 'cfs': MASTER_CFS, 'mesa': MASTER_MESA, 'sof': MASTER_SOF, 'hsp': MASTER_HSP}


RAW_SHHS = [
    META_PATH + "/" + SHHS + "/datasets/shhs1-dataset-0.21.0.csv",
    META_PATH + "/" + SHHS + "/datasets/shhs2-dataset-0.21.0.csv",
]

RAW_MROS = [
    META_PATH + "/" + MROS + "/datasets/mros-visit1-dataset-0.6.0.csv",
    META_PATH + "/" + MROS + "/datasets/mros-visit2-dataset-0.6.0.csv",
]

RAW_CFS = [
    META_PATH + "/" + CFS + "/datasets/cfs-visit5-dataset-0.7.0.csv",
]

RAW_MESA = [
    META_PATH + "/" + MESA + "/datasets/mesa-sleep-dataset-0.7.0.csv",
]

RAW_SOF = [
    META_PATH + "/" + SOF + "/datasets/sof-visit-8-dataset-0.8.0.csv",
]
##### unified preprocessing setting
# define for preprocessing


ECG = 'ECG'
ECG1 = 'ECG1'
ECG2 = 'ECG2'
ECG3 = 'ECG3'
HR = 'HR'
PPG = 'PPG'

SPO2 = 'SPO2'
OX = 'OX'
ABD = 'ABD'
THX = 'THX'
AF = 'AF'
NP = 'NP'
SN = 'SN'
PPG = 'PPG'

EOG_L = 'EOG_L'
EOG_R = 'EOG_R'
EOG_E1_A2 = 'EOG_E1_A2'
EOG_E2_A1 = 'EOG_E2_A1'

EMG_LLeg = 'EMG_LLeg' # EMG_LLeg1 - EMG_LLeg2
EMG_RLeg = 'EMG_RLeg' # EMG_RLeg1 - EMG_RLeg2
EMG_LLeg1 = 'EMG_LLeg1'
EMG_LLeg2 = 'EMG_LLeg2'
EMG_RLeg1 = 'EMG_RLeg1'
EMG_RLeg2 = 'EMG_RLeg2'

EMG_Chin = 'EMG_Chin' # bipolar
EMG_RChin = 'EMG_RChin'
EMG_LChin = 'EMG_LChin'
EMG_CChin = 'EMG_CChin'

EEG_C3 = 'EEG_C3'
EEG_C4 = 'EEG_C4'
EEG_A1 = 'EEG_A1'
EEG_A2 = 'EEG_A2'
EEG_O1 = 'EEG_O1'
EEG_O2 = 'EEG_O2'
EEG_F3 = 'EEG_F3'
EEG_F4 = 'EEG_F4'

EEG_C3_A2 = 'EEG_C3_A2'
EEG_C4_A1 = 'EEG_C4_A1'
EEG_F3_A2 = 'EEG_F3_A2'
EEG_F4_A1 = 'EEG_F4_A1'
EEG_O1_A2 = 'EEG_O1_A2'
EEG_O2_A1 = 'EEG_O2_A1'

FPZ = 'FPZ' # not in output but potentially useful to define channels relative to ground
GROUND = 'GROUND' # not in output but potentially useful to define channels relative to ground

POS = 'POS'

FREQ_ECG = 128
FREQ_ECG1 = 128
FREQ_ECG2 = 128
FREQ_ECG3 = 128
FREQ_HR = 1
FREQ_PPG = 128

FREQ_SPO2 = 1
FREQ_OX = 1 # Ox status
FREQ_ABD = 8
FREQ_THX = 8
FREQ_AF = 8 # airflow
FREQ_NP = 8 # Nasal Pressure
FREQ_SN = 32 # snore
# FREQ_PPG = 8 # chat: 10~100 hz or maybe higher, others 128~256

FREQ_EOG_L = 64
FREQ_EOG_R = 64

FREQ_EOG_E1_A2 = 64
FREQ_EOG_E2_A1 = 64

FREQ_EMG_LLeg = 64
FREQ_EMG_RLeg = 64
FREQ_EMG_LLeg1 = 64
FREQ_EMG_LLeg2 = 64
FREQ_EMG_RLeg1 = 64
FREQ_EMG_RLeg2 = 64

FREQ_EMG_Chin = 64
FREQ_EMG_LChin = 64
FREQ_EMG_RChin = 64
FREQ_EMG_CChin = 64

FREQ_EEG_C3 = 64
FREQ_EEG_C4 = 64
FREQ_EEG_A1 = 64
FREQ_EEG_A2 = 64
FREQ_EEG_O1 = 64
FREQ_EEG_O2 = 64
FREQ_EEG_F3 = 64
FREQ_EEG_F4 = 64

FREQ_EEG_C3_A2 = 64
FREQ_EEG_C4_A1 = 64
FREQ_EEG_F3_A2 = 64
FREQ_EEG_F4_A1 = 64
FREQ_EEG_O1_A2 = 64
FREQ_EEG_O2_A1 = 64

FREQ_POS = 1



EDF_COLS = [ECG, ECG1, ECG2, ECG3, HR, PPG,
            SPO2, OX, ABD, THX, AF, NP, SN, 
            EOG_L, EOG_R, EOG_E1_A2, EOG_E2_A1,
            EMG_LLeg, EMG_RLeg, EMG_LLeg1, EMG_LLeg2, EMG_RLeg1, EMG_RLeg2, EMG_Chin, EMG_RChin, EMG_LChin, EMG_CChin,

            EEG_C3, EEG_C4, EEG_A1, EEG_A2, EEG_O1, EEG_O2, EEG_F3, EEG_F4,
            EEG_C3_A2, EEG_C4_A1, EEG_F3_A2, EEG_F4_A1, EEG_O1_A2, EEG_O2_A1,
            POS]

# NOTE: why is this set to 8 hours? should we really standardize the length of all recordings?
# TODO: change to not standardize the length of all recordings?
MAX_LENGTH = 60 * 60 * 12  # Recording length in seconds (trimmed to 10h)
TARGET_LABEL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1, 30.0)[1:])

ECG_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_ECG)[1:])
ECG1_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_ECG1)[1:])
ECG2_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_ECG2)[1:])
ECG3_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_ECG3)[1:])
HR_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_HR)[1:])
PPG_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_PPG)[1:])

SPO2_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_SPO2)[1:])
OX_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_OX)[1:])
ABD_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_ABD)[1:])
THX_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_THX)[1:])
AF_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_AF)[1:])
NP_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_NP)[1:])
SN_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_SN)[1:])


EMG_LLeg_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_EMG_LLeg)[1:])
EMG_RLeg_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_EMG_RLeg)[1:])
EMG_LLeg1_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_EMG_LLeg1)[1:])
EMG_LLeg2_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_EMG_LLeg2)[1:])
EMG_RLeg1_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_EMG_RLeg1)[1:])
EMG_RLeg2_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_EMG_RLeg2)[1:])

EMG_Chin_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_EMG_Chin)[1:])
EMG_LChin_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_EMG_LChin)[1:])
EMG_RChin_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_EMG_RChin)[1:])
EMG_CChin_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_EMG_CChin)[1:])

EOG_L_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_EOG_L)[1:])
EOG_R_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_EOG_R)[1:])
EOG_E1_A2_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_EOG_E1_A2)[1:])
EOG_E2_A1_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_EOG_E2_A1)[1:])

EEG_C3_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_EEG_C3)[1:])
EEG_C4_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_EEG_C4)[1:])
EEG_A1_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_EEG_A1)[1:])
EEG_A2_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_EEG_A2)[1:])
EEG_O1_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_EEG_O1)[1:])
EEG_O2_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_EEG_O2)[1:])
EEG_F3_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_EEG_F3)[1:])
EEG_F4_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_EEG_F4)[1:])

EEG_C3_A2_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_EEG_C3_A2)[1:])
EEG_C4_A1_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_EEG_C4_A1)[1:])
EEG_F3_A2_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_EEG_F3_A2)[1:])
EEG_F4_A1_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_EEG_F4_A1)[1:])
EEG_O1_A2_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_EEG_O1_A2)[1:])
EEG_O2_A1_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_EEG_O2_A1)[1:])

POS_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 1 / FREQ_POS)[1:])

##### dataset-specific 1: clinical label extraction

# --- HSP ---
HSP_S0001_CLINICAL_VARS = [
 'BMI', 
 'ESS',
 'FHxSleepDisorder', 
 'alcoholHelpSleep', 
 'caffeinatedDrinksNum', 
 'dxAFib',
 'dxAbnormalHeartRhythm',
 'dxAnxiety',
 'dxApneaTried_cpapBipap',
 'dxApneaTried_dentalDevice',
 'dxApneaTried_gastricBypassSurgery',
 'dxApneaTried_positionTherapy',
 'dxApneaTried_surgery',
 'dxApneaTried_weightLoss',
 'dxArthritis',
 'dxAsthmaExerciseInduced',
 'dxBipolarDisorder',
 'dxBronchitis',
 'dxCOPDOrEmphysema',
 'dxCancer',
 'dxCongestiveHeartFailure',
 'dxCoronaryDisease',
 'dxDepression',
 'dxDiabetes',
 'dxFibromyalgia',
 'dxHeadTrauma',
 'dxHeadaches',
 'dxHeartAttack',
 'dxHighBloodPressure',
 'dxHyperthyroid',
 'dxHypothyroid',
 'dxKidneyDisease',
 'dxKindOfCancer',
 'dxMemoryProblems',
 'dxMeningitis',
 'dxPTSD',
 'dxPacemaker',
 'dxRestlessLegs',
 'dxSeizures',
 'dxSleepApnea',
 'dxStroke', 
 'haveAllergies',
 'height', 
 'legsFeelFunny', 
 'musclesGetWeakLaughOrAngry', 
 'packsPerDay', 
 'troubleFallingAsleep',
 'troubleStayingAsleep', 
 'weight'
 ]

# ---

NA_FLAG = '1foaushf08hg'
ALT_COLUMNS = { # for pre-processing time
    ECG: ( # preprocessed, ECG3 - ECG1 or ECG2 - ECG1
        'ECG', # shhs
        NA_FLAG, # chat
        NA_FLAG, # mros
        NA_FLAG, # ccshs
        NA_FLAG, # cfs 
        'EKG', # mesa
        NA_FLAG, # sof
        'ECG', # numom2b
        'ECG', # wsc
        'EKG', # hsp

        'ECG EKG2-EKG', # nchsdb

        'ECG II', # hsp
        'ECG', # hsp
        'EKG_EG', # hsp
        'Bipolar 2', # hsp

         ),
    ECG1: ( # right clavicle or just right
        NA_FLAG, # shhs
        'ECG1', # chat
        'ECG R', # mros
        'ECG1', # ccshs
        'ECG1', # cfs
        NA_FLAG, # mesa
        'ECG1', # sof
        NA_FLAG, # numom2b

        NA_FLAG, # wsc
        NA_FLAG, # nchsdb

        'ECGR', # hsp
        'EKG-R', # hsp
        #'ECG-V1', # hsp (NOTE: in 12-lead system, ECG-RA also exists)
        'ECG-RA', # hsp

    ),
    ECG2: ( # left clavicle or just left
        NA_FLAG, # shhs
        'ECG2', # chat
        'ECG L', # mros
        'ECG2', # ccshs
        'ECG2', # cfs
        NA_FLAG, # mesa
        'ECG2', # sof
        NA_FLAG, # numom2b

        NA_FLAG, # wsc
        NA_FLAG, # nchsdb
        

        'ECGL', # hsp
        'EKG-L', # hsp
        #'ECG-V2', # hsp (NOTE: in 12-lead system, ECG-LA also exists)
        'ECG-LA', # hsp

    ),
    ECG3: ( # left leg (standard) or left lower rib (modified)
        NA_FLAG, # shhs
        'ECG3', # chat
        NA_FLAG, # mros
        NA_FLAG, # ccshs
        NA_FLAG, # cfs
        NA_FLAG, # mesa
        NA_FLAG, # sof
        NA_FLAG, # numom2b

        NA_FLAG, # wsc
        NA_FLAG, # nchsdb
    

        'ECG-LL', # hsp

    ),
    HR: (
        'H.R.', # shhs
        'PR', # shhs2
        NA_FLAG, # chat -> NA
        'HR', # mros
        'HRate', # ccshs
        'HRate', # cfs
        'HR', # mesa
        'HR', # sof

        NA_FLAG, # wsc
        NA_FLAG, # nchsdb

        'PR', # hsp
        'Pulse', # hsp (NOTE: think about if pulse can be grouped with HR)
        'Heart Rate_DR', # hsp
        'HR', # hsp
        'PULSE', # hsp
        'PulseRate', # hsp
        'Pulse_EG', # hsp
    ),
    PPG: (
        NA_FLAG, # shhs
        NA_FLAG, # chat
        NA_FLAG, # mros
        NA_FLAG, # ccshs
        NA_FLAG, # cfs
        NA_FLAG, # mesa
        NA_FLAG, # sof
        'Pleth', # hsp
        'Plethysmogram', # hsp 
        'PPG', # hsp
        'Finger PPG', # hsp
        'PlethWV', # ccshs

    ),
    SPO2: (
        'SaO2', # shhs
        'SaO2', # chat
        'SaO2', # mros
        'SpO2', # ccshs
        'SpO2', # cfs
        'SpO2', # mesa
        'ASO2', # sof
        'SaO2', # hsp
        'SpO2', # hsp

        'spo2', # wsc
        'SpO2', # nchsdb

        'OSAT', # hsp
        'SpO2 BB', # hsp (NOTE: what is BB?)
        'SpO2_EG', # hsp (NOTE: what is EG?)
        'SpO2-BB_EG', # hsp (NOTE: what is BB?)
        'SPO2 B-B', # hsp (NOTE: what is B-B?)
        'SPO2', # hsp

    ),
    OX: (
        'OX stat', # shhs
        'OX STAT', # shhs2
        NA_FLAG, # chat -> NA
        'STAT', # mros
        'OX STATUS', # ccshs
        'OX STATUS', # cfs
        'OxStatus', # mesa
        'STAT', # sof

        NA_FLAG, # wsc
        NA_FLAG, # nchsdb

        NA_FLAG, # hsp

    ),
    ABD: (
        'ABDO RES', # shhs
        'ABD', # chat
        'Abdominal', # mros
        'ABDO EFFORT', # ccshs
        'ABDO EFFORT', # cfs
        'Abdo', # mesa
        'Abdominal', # sof
        'ABDOMINAL', # hsp
        'ABD', # hsp
        'Abdomen', # hsp

        'abdomen', # wsc
        'Resp Abdominal', # nchsdb

        'Abdominal Effort', # hsp
        'Effort ABD', # hsp
        'Abdomen_EG', # hsp
        'ABDOMEN', # hsp
        'Abd', # hsp
        'ABDOM', # hsp

    ),
    THX: (
        'THOR RES', # shhs
        'Chest', # chat
        'Thoracic', # mros
        'THOR EFFORT', # ccshs
        'THOR EFFORT', # cfs
        'Thor', # mesa
        'Thoracic', # sof

        'CHEST', # hsp
        'thorax', # wsc
        'Resp Thoracic', # nchsdb

        'CHEST', # hsp
        'Thorax', # hsp
        'Chest', # hsp
        'THORACIC', # hsp
        'Thoracic Effort', # hsp
        'Effort THO', # hsp
        'Thorax_EG', # hsp
        'THORAX', # hsp
        'THOR', # hsp

    ),
    AF: (
        'AIRFLOW', # shhs
        'Airflow', # chat
        'Airflow', # mros
        'AIRFLOW', # ccshs
        'AIRFLOW', # cfs
        NA_FLAG, # mesa -> NA
        'Airflow', # sof

        'flow', # wsc
        'Resp Airflow', # nchsdb

        'Flow_DR', # hsp (NOTE: also not sure about this, no docs for hsp, I asked for more info on channels)
        'PTAF', # hsp
        'FLOW', # hsp (NOTE: not entirely sure if this is airflow or cannula flow)
        'AIRFLOW', # hsp
        'AirFlow', # hsp
        'Flow', # hsp
        'Airflow2', # hsp (NOTE: make sure no duplicates in final df?)
        'Airflow', # hsp
        'Flow_EG', # hsp (NOTE: what is EG?)

    ),
    NP: (
        NA_FLAG, # shhs -> NA
        'CannulaFlow', # chat
        'Cannula Flow', # mros
        'NASAL PRES', # ccshs
        'NASAL PRES', # cfs
        'Flow', # mesa
        'Nasal Pressure', # sof
        'C-FLOW', # hsp (NOTE: check if this corresponds to CPAP titration or treatment, or just short for cannula flow)

        'CFLOW', # hsp
        'nas_pres', # wsc
        NA_FLAG, # nchsdb

        'CFLOW', # hsp (NOTE: check if this corresponds to CPAP titration or treatment, or just short for cannula flow)
        'Nasal Pressure', # hsp
        'Nasal', # hsp
        'CFlow', # hsp (NOTE: check if this corresponds to CPAP titration or treatment, or just short for cannula flow)
        'Cannula', # hsp
        'Nasal Canula', # hsp
        'Nasal_EG', # hsp (NOTE: what is EG?)

    ),
    SN: (
        NA_FLAG, # shhs -> NA
        'Snore', # chat
        NA_FLAG, # mros -> NA
        'SNORE', # ccshs
        'SNORE', # cfs
        'Snore', # mesa
        NA_FLAG, # sof -> NA
        'SNORE', # hsp
        'snore', # wsc
        'Snore_DR', # hsp (NOTE: what is DR?)
        'Snore', # hsp
        'Snoring Sensor', # hsp
        'Snore_EG', # hsp (NOTE: what is EG?)
        'Snoring', # hsp

    ),
    EOG_L: (
        'EOG(L)', # shhs
        'E1', # chat
        'EOG L', # mros
        'LOC', # ccshs
        'LOC', # cfs
        'EOG-L_Off', # mesa
        'LOC', # sof
        'E1', # hsp

        NA_FLAG, # wsc
        NA_FLAG, # nchsdb

        'LOC', # hsp
        
        'L-EOG', # hsp

    ),
    EOG_R: (
        'EOG(R)', # shhs
        'E2', # chat
        'EOG R', # mros
        'ROC', # ccshs
        'ROC', # cfs
        'EOG-R_Off', # mesa
        'ROC', # sof
        'E2', # hsp

        NA_FLAG, # wsc
        NA_FLAG, # nchsdb
    ),
    EOG_E1_A2: (
        NA_FLAG, # shhs
        NA_FLAG, # chat
        NA_FLAG, # mros
        NA_FLAG, # ccshs
        NA_FLAG, # cfs
        NA_FLAG, # mesa
        NA_FLAG, # sof
        'E1', # wsc
        'EOG LOC-M2', # nchsdb
        'E1-M2', # hsp
        'LOC-A2', # hsp
        'EOG LOC-A2', # hsp
    ),
    EOG_E2_A1: (
        NA_FLAG, # shhs
        NA_FLAG, # chat
        NA_FLAG, # mros
        NA_FLAG, # ccshs
        NA_FLAG, # cfs
        NA_FLAG, # mesa
        NA_FLAG, # sof
        'E2', # wsc
        'EOG ROC-M1', # nchsdb
        'ROC', # hsp
        'E2-M1', # hsp
        'E2-M2', # hsp (NOTE: different reference M2 but still E2 signal)
        'ROC-A1', # hsp
        'EOG ROC-A2', # hsp
        'R-EOG', # hsp

    ),
    # TODO: mros and ccshs leg channels move to lleg1 lleg2 rleg1 rleg2? (check with Zitao?)
    EMG_LLeg: (
        NA_FLAG, # shhs -> NA
        NA_FLAG, # chat
        'Leg L', # mros
        NA_FLAG, # ccshs
        'L Leg', # cfs
        NA_FLAG, # mesa -> NA
        'Leg/L', # sof
        'LAT', # hsp

        NA_FLAG, # nchsdb

        'Left Leg', # hsp
        'Leg', # hsp
        'EMG-Leg', # hsp
        'L LEG', # hsp

    ),
    EMG_RLeg: (
        NA_FLAG, # shhs -> NA
        NA_FLAG, # chat
        'Leg R', # mros
        NA_FLAG, # ccshs
        'R Leg', # cfs
        NA_FLAG, # mesa -> NA
        'Leg/R', # sof
        'RAT', # hsp

        NA_FLAG, # nchsdb

        'Right Leg', # hsp
        'R LEG', # hsp
    ),
    EMG_LLeg1: (
        NA_FLAG, # shhs
        'Lleg1', # chat
        NA_FLAG, # mros
        'LEFT LEG1', # ccshs
        NA_FLAG, # cfs
        NA_FLAG, # mesa
        NA_FLAG, # sof
        'LAT1', # hsp
        'LAT-U', # hsp
        'LLEG+', # hsp
        'Leg 1', # hsp
    ),
    EMG_LLeg2: (
        NA_FLAG, # shhs
        'Lleg2', # chat
        NA_FLAG, # mros
        'LEFT LEG2', # ccshs
        NA_FLAG, # cfs
        NA_FLAG, # mesa
        NA_FLAG, # sof
        'LAT2', # hsp
        'LAT-L', # hsp
        'LLEG-', # hsp
        'Leg 2', # hsp
    ),
    EMG_RLeg1: (
        NA_FLAG, # shhs
        'Rleg1', # chat
        NA_FLAG, # mros
        'RIGHT LEG1', # ccshs
        NA_FLAG, # cfs
        NA_FLAG, # mesa
        NA_FLAG, # sof
        'RAT1', # hsp
        'RAT-U', # hsp
        'RLEG+', # hsp
    ),
    EMG_RLeg2: (
        NA_FLAG, # shhs
        'Rleg2', # chat
        NA_FLAG, # mros
        'RIGHT LEG2', # ccshs
        NA_FLAG, # cfs
        NA_FLAG, # mesa
        NA_FLAG, # sof
        'RAT2', # hsp
        'RAT-L', # hsp
        'RLEG-', # hsp
    ),
    EMG_Chin: (
        'EMG', # shhs
        NA_FLAG, # chat
        NA_FLAG, # mros
        NA_FLAG, # ccshs
        NA_FLAG, # cfs
        'EMG', # mesa
        NA_FLAG, # sof
        'CHIN', # hsp
        'CHIN1-CHIN2', # hsp  (NOTE: depends on 1, 2, 3, configuration, but count as bipolar)
        'Chin1-Chin2', # hsp  (NOTE: depends on 1, 2, 3, configuration, but count as bipolar)
        'Chin EMG', # hsp
        'EMG Chin', # hsp
        'EMG-Chin', # hsp
        'Chin', # hsp
        'Chin1-Chin3', # hsp (NOTE: depends on 1, 2, 3, configuration, but count as bipolar)
        'CHIN2-CHIN3', # hsp (NOTE: depends on 1, 2, 3, configuration, but count as bipolar)
    ),
    EMG_LChin: (
        NA_FLAG, # shhs-> NA
        'Lchin', # chat
        'L Chin', # mros
        'EMG2', # ccshs  (NOTE: double check documentation)
        'EMG2', # cfs  (NOTE: double check documentation)
        NA_FLAG, # mesa
        'L Chin', # sof
        'CHIN2', # hsp (NOTE: not sure because there is no extra documentation, but other datasets use 2 for left chin)
        'ChinL', # hsp
        'Chin-L', # hsp
        'Chin2', # hsp (NOTE: not sure because there is no extra documentation, but other datasets use 2 for left chin)
    ),
    EMG_RChin: (
        NA_FLAG, # shhs -> NA
        'Rchin', # chat
        'R Chin', # mros
        'EMG3', # ccshs  (NOTE: double check documentation)
        'EMG3', # cfs  (NOTE: double check documentation)
        NA_FLAG, # mesa
        'R Chin', # sof
        'CHIN3', # hsp (NOTE: not sure if this is right, but other datasets use 3 for right chin)
        'ChinR', # hsp
        'Chin-R', # hsp
        'Chin3', # hsp (NOTE: not sure if this is right, but other datasets use 3 for right chin)
    ),
    EMG_CChin: (
        NA_FLAG, # shhs
        'Cchin', # chat
        NA_FLAG, # mros
        'EMG1', # ccshs (NOTE: double check documentation)
        'EMG1', # cfs (NOTE: double check documentation)
        NA_FLAG, # mesa
        NA_FLAG, # sof
        'CHIN1', # hsp (NOTE: not sure because there is no extra documentation, 1 indicates cchin for other datasets)
        'ChinA', # hsp (NOTE: not entirely sure if this is center, but R and L also exist so it must be center?)
        'CHINz', # hsp (z probably indicates center like for EEG?)
        'Chin-Ctr', # hsp
    ),
    EEG_C3: (
        NA_FLAG, # shhs
        'C3', # chat
        'C3', # mros
        'C3', # ccshs
        'C3', # cfs
        '', # mesa -> NA
        'C3', # sof -> NA
        'C3', # hsp
        NA_FLAG, # nchsdb
        NA_FLAG, # wsc
    ),
    EEG_C4: (
        NA_FLAG, # shhs
        'C4', # chat
        'C4', # mros
        'C4', # ccshs
        'C4', # cfs
        '', # mesa -> NA
        'C4', # sof -> NA
        'C4', # hsp
        NA_FLAG, # nchsdb
        NA_FLAG, # wsc
    ),
    EEG_A1: (
        NA_FLAG, # shhs -> NA
        'M1', # chat
        'A1', # mros
        'A1', # ccshs
        NA_FLAG, # cfs
        NA_FLAG, # mesa
        NA_FLAG, # sof
        'M1', # hsp

        NA_FLAG, # nchsdb
        NA_FLAG, # wsc

        'A1', # hsp

    ),
    EEG_A2: (
        NA_FLAG, # shhs -> NA
        'M2', # chat
        'A2', # mros
        'A2', # ccshs
        NA_FLAG, # cfs
        NA_FLAG, # mesa
        NA_FLAG, # sof
        'M2', # hsp

        NA_FLAG, # nchsdb
        NA_FLAG, # wsc

        'A2', # hsp

    ),
    EEG_O1: (
        NA_FLAG, # shhs -> NA
        'O1', # chat
        NA_FLAG, # mros
        NA_FLAG, # ccshs
        NA_FLAG, # cfs
        NA_FLAG, # mesa
        NA_FLAG, # sof
        'O1', # hsp
        NA_FLAG, # nchsdb
        NA_FLAG, # wsc
    ),
    EEG_O2: (
        NA_FLAG, # shhs -> NA
        'O1', # chat
        NA_FLAG, # mros
        NA_FLAG, # ccshs
        NA_FLAG, # cfs
        NA_FLAG, # mesa
        NA_FLAG, # sof
        'O2', # hsp
        NA_FLAG, # nchsdb
        NA_FLAG, # wsc
    ),
    EEG_F3: (
        NA_FLAG, # shhs -> NA
        'F3', # chat
        NA_FLAG, # mros
        NA_FLAG, # ccshs
        NA_FLAG, # cfs
        NA_FLAG, # mesa
        NA_FLAG, # sof
        'F3', # hsp
        NA_FLAG, # nchsdb
        NA_FLAG, # wsc
    ),
    EEG_F4: (
        NA_FLAG, # shhs -> NA
        'F4', # chat
        NA_FLAG, # mros
        NA_FLAG, # ccshs
        NA_FLAG, # cfs
        NA_FLAG, # mesa
        NA_FLAG, # sof
        'F4', # hsp
        NA_FLAG, # nchsdb
        NA_FLAG, # wsc
    ),
    EEG_C3_A2: (
        'EEG(sec)', # shhs
        NA_FLAG, # chat
        NA_FLAG, # mros
        NA_FLAG, # ccshs
        NA_FLAG, # cfs
        NA_FLAG, # mesa
        NA_FLAG, # sof
        'C3-M2', # hsp

        'EEG C3-M2', # nchsdb
        'C3_M2', # wsc

        'C3-A2', # hsp
        'EEG C3-A2', # hsp

    ),
    EEG_C4_A1: (
        'EEG', # shhs
        NA_FLAG, # chat
        NA_FLAG, # mros
        NA_FLAG, # ccshs
        NA_FLAG, # cfs
        'EEG3', # mesa
        NA_FLAG, # sof
        'C4-M1', # hsp

        'EEG C4-M1', # nchsdb
        NA_FLAG, # wsc

        'C4-A1', # hsp
        'EEG C4-A1', # hsp

    ),
    EEG_F3_A2: (
        NA_FLAG, # shhs -> NA
        NA_FLAG, # chat
        NA_FLAG, # mros
        NA_FLAG, # ccshs
        NA_FLAG, # cfs
        NA_FLAG, # mesa
        NA_FLAG, # sof
        'F3-M2', # hsp

        'EEG F3-M2', # nchsdb
        'F3_M2', # wsc

        'EEG F3-A2', # hsp

    ),
    EEG_F4_A1: (
        NA_FLAG, # shhs -> NA
        NA_FLAG, # chat
        NA_FLAG, # mros
        NA_FLAG, # ccshs
        NA_FLAG, # cfs
        NA_FLAG, # mesa
        NA_FLAG, # sof
        'F4-M1', # hsp

        'EEG F4-M1', # nchsdb
        NA_FLAG, # wsc

        'EEG F4-A1', # hsp

    ),
    EEG_O1_A2: (
        NA_FLAG, # shhs -> NA
        NA_FLAG, # chat
        NA_FLAG, # mros
        NA_FLAG, # ccshs
        NA_FLAG, # cfs
        NA_FLAG, # mesa
        NA_FLAG, # sof
        'O1-M2', # hsp

        'EEG O1-M2', # nchsdb
        'O1_M2', # wsc

        'O1-A2', # hsp
        'EEG O1-A2', # hsp

    ),
    EEG_O2_A1: (
        NA_FLAG, # shhs -> NA
        NA_FLAG, # chat
        NA_FLAG, # mros
        NA_FLAG, # ccshs
        NA_FLAG, # cfs
        NA_FLAG, # mesa
        NA_FLAG, # sof
        'O2-M1', # hsp

        'EEG O2-M1', # nchsdb
        NA_FLAG, # wsc

        'O2-A1', # hsp
        'EEG O2-A1', # hsp

    ),
    FPZ: (
        NA_FLAG, # shhs -> NA
        NA_FLAG, # chat
        NA_FLAG, # mros
        NA_FLAG, # ccshs
        NA_FLAG, # cfs
        NA_FLAG, # mesa
        NA_FLAG, # sof
        'Fpz', # hsp (NOTE: slightly different from FPZ but still midline sagittal plane)
    ),
    GROUND: (
        NA_FLAG, # shhs -> NA
        NA_FLAG, # chat
        NA_FLAG, # mros
        NA_FLAG, # ccshs
        NA_FLAG, # cfs
        NA_FLAG, # mesa
        NA_FLAG, # sof

        NA_FLAG, # wsc
    ),
    POS: (
        'Position', # shhs
        'Position', # chat
        'Position', # mros
        'Pos', # mesa
        'POSITION', # ccshs
        'Position', # cfs
        'Position', # sof
        'position', # wsc
        'Position', # hsp
        NA_FLAG, # nchsdb
        NA_FLAG, # wsc
        'Position', # shhs
        NA_FLAG, # chat
        NA_FLAG, # mros
        NA_FLAG, # ccshs
        NA_FLAG, # cfs
        NA_FLAG, # mesa
        NA_FLAG, # sof
        'Position', # hsp
        'POS', # hsp
        'Position_DR', # hsp
        'Position_EG', # hsp

    ),
}

# TODO: add PPG after merging with Zitao
# TODO: change to EMG_Chin
# TODO: just create an indicator for which columns are available for every night during processing? 
CHANNEL_FLAG_UNIMOD = { # for training time filtering

    SHHS: {
        ECG: 1,
        HR: 1,
        SPO2: 1,
        OX: 1,
        ABD: 1,
        THX: 1,
        AF: 1,
        NP: 0,
        SN: 0,
        PPG: 0,
        EOG_E1_A2: 0,
        EOG_E2_A1: 0,
        EMG_LLeg: 0,
        EMG_RLeg: 0,
        EMG_LChin: 0,
        EMG_RChin: 0,
        EMG_CChin: 1,
        EEG_C3_A2: 1,
        EEG_C4_A1: 1,
        EEG_F3_A2: 0,
        EEG_F4_A1: 0,
        EEG_O1_A2: 0,
        EEG_O2_A1: 0,
        POS: 1
    },
    CHAT: {
        ECG: 1,
        HR: 0,
        SPO2: 1,
        OX: 0,
        ABD: 1,
        THX: 1,
        AF: 1,
        NP: 1,
        SN: 1,
        PPG: 1,
        EOG_E1_A2: 1,
        EOG_E2_A1: 1,
        EMG_LLeg: 1,
        EMG_RLeg: 1,
        EMG_LChin: 1,
        EMG_RChin: 1,
        EMG_CChin: 1,
        EEG_C3_A2: 1,
        EEG_C4_A1: 1,
        EEG_F3_A2: 1,
        EEG_F4_A1: 1,
        EEG_O1_A2: 1,
        EEG_O2_A1: 1,
        POS: 1
    },
    MROS: {
        ECG: 1,
        HR: 1,
        SPO2: 1,
        OX: 1,
        ABD: 1,
        THX: 1,
        AF: 1,
        NP: 1,
        SN: 0,
        PPG: 0,
        EOG_E1_A2: 1,
        EOG_E2_A1: 1,
        EMG_LLeg: 1,
        EMG_RLeg: 1,
        EMG_LChin: 1,
        EMG_RChin: 1,
        EMG_CChin: 0,
        EEG_C3_A2: 1,
        EEG_C4_A1: 1,
        EEG_F3_A2: 0,
        EEG_F4_A1: 0,
        EEG_O1_A2: 0,
        EEG_O2_A1: 0,
        POS: 1
    },
    
    CCSHS: {
        ECG: 1,
        HR: 1,
        SPO2: 1,
        OX: 1,
        ABD: 1,
        THX: 1,
        AF: 1,
        NP: 1,
        SN: 1,
        PPG: 1,
        EOG_E1_A2: 1,
        EOG_E2_A1: 1,
        EMG_LLeg: 1,
        EMG_RLeg: 1,
        EMG_LChin: 1,
        EMG_RChin: 1,
        EMG_CChin: 1,
        EEG_C3_A2: 1,
        EEG_C4_A1: 1,
        EEG_F3_A2: 0,
        EEG_F4_A1: 0,
        EEG_O1_A2: 0,
        EEG_O2_A1: 0,
        POS: 1
    },
    CFS: {
        ECG: 1,
        HR: 0,
        SPO2: 1,
        OX: 1,
        ABD: 1,
        THX: 1,
        AF: 1,
        NP: 1,
        SN: 1,
        PPG: 0,
        EOG_E1_A2: 1,
        EOG_E2_A1: 1,
        EMG_LLeg: 1,
        EMG_RLeg: 1,
        EMG_LChin: 1,
        EMG_RChin: 1,
        EMG_CChin: 1,
        EEG_C3_A2: 1,
        EEG_C4_A1: 1,
        EEG_F3_A2: 0,
        EEG_F4_A1: 0,
        EEG_O1_A2: 0,
        EEG_O2_A1: 0,
        POS: 1
    },
    MESA: {
        ECG: 1,
        HR: 1,
        SPO2: 1,
        OX: 1,
        ABD: 1,
        THX: 1,
        AF: 0,
        NP: 1,
        SN: 1,
        PPG: 1,
        EOG_E1_A2: 1,
        EOG_E2_A1: 1,
        EMG_LLeg: 0,
        EMG_RLeg: 0,
        EMG_LChin: 0,
        EMG_RChin: 0,
        EMG_CChin: 1,
        EEG_C3_A2: 0,
        EEG_C4_A1: 1,
        EEG_F3_A2: 0,
        EEG_F4_A1: 0,
        EEG_O1_A2: 0,
        EEG_O2_A1: 0,
        POS: 1
    },
    SOF: {
        ECG: 1,
        HR: 1,
        SPO2: 1,
        OX: 1,
        ABD: 1,
        THX: 1,
        AF: 1,
        NP: 1,
        SN: 0,
        PPG: 0,
        EOG_E1_A2: 1,
        EOG_E2_A1: 1,
        EMG_LLeg: 1,
        EMG_RLeg: 1,
        EMG_LChin: 1,
        EMG_RChin: 1,
        EMG_CChin: 0,
        EEG_C3_A2: 1,
        EEG_C4_A1: 1,
        EEG_F3_A2: 0,
        EEG_F4_A1: 0,
        EEG_O1_A2: 0,
        EEG_O2_A1: 0,
        POS: 1
    },
    HSP: {} # variable between people even within the same site, will make channel indicator in processing script
}

##### Standard column names for annotations
# we extract the event name, start time in seconds, and end time in seconds. 
EVENT_NAME_COLUMN = 'EVENT'
START_TIME_COLUMN = 'START_SEC'
END_TIME_COLUMN = 'END_SEC'


##### dataset-specific 3: event annotation groupings and alternative names 

## Event name configs

# Sleep stage integer labels, careful if mapping from vectorized integers to these labels
SLEEP_STAGE_UNKNOWN = -1 # 'Unknown'
SLEEP_STAGE_WAKE = 0 # 'Wake'
SLEEP_STAGE_LIGHT_SLEEP = 1 # 'Light Sleep'
SLEEP_STAGE_DEEP_SLEEP = 2 # 'Deep Sleep'
SLEEP_STAGE_REM_SLEEP = 3 # 'REM'

# Respiratory event names
RESPIRATORY_EVENT_CENTRAL_APNEA = 'Central Apnea'
RESPIRATORY_EVENT_OBSTRUCTIVE_APNEA = 'Obstructive Apnea'
RESPIRATORY_EVENT_MIXED_APNEA = 'Mixed Apnea'
RESPIRATORY_EVENT_HYPOPNEA = 'Hypopnea'
RESPIRATORY_EVENT_DESATURATION = 'Oxygen Desaturation'

# Limb event names 
LIMB_MOVEMENT_ISOLATED = 'Limb Movement Isolated'
LIMB_MOVEMENT_PERIODIC = 'Limb Movement Periodic'
LIMB_MOVEMENT_ISOLATED_LEFT = 'Left Limb Movement Isolated'
LIMB_MOVEMENT_ISOLATED_RIGHT = 'Right Limb Movement Isolated'
LIMB_MOVEMENT_PERIODIC_LEFT = 'Left Limb Movement Periodic'
LIMB_MOVEMENT_PERIODIC_RIGHT = 'Right Limb Movement Periodic'

# Arousal event names
AROUSAL_EVENT_CLASSIC = 'Arousal' # this is classic ASDA arousal in many sleep studies, describing a shift in EEG frequency
AROUSAL_EVENT_RESPIRATORY = 'RERA'
AROUSAL_EVENT_EMG = 'EMG-Related Arousal'

# Lights on/off event names
LIGHTS_ON = 'Lights On'
LIGHTS_OFF = 'Lights Off'

## Event name mapping configs
# TODO: we need to standardize event names across all other datasets too
EVENT_NAME_MAPPING = {

    # Sleep Stage Groupings
    'Sleep Stages': { # these are consistent across nsrr datasets
        SLEEP_STAGE_UNKNOWN: (
            'Unknown',
            '?',
            'Unscored', 
            'Unscored|9', 
            'Movement|6', # store movements as unknown for now (as in wav2sleep)
            9,
            6, 
            7, # not sure why, but this is mapped to np.nan in wav2sleep so I will keep it here too
            'unknownstage', # hsp I0004
        ),
        SLEEP_STAGE_WAKE: (
            'Wake',
            'Awake',
            'Wake|0', 
            0, 
            'wake', # hsp I0004
        ),
        SLEEP_STAGE_LIGHT_SLEEP: (
            'Light Sleep',
            'Stage_1',
            'N1',
            'Stage_2',
            'N2',
            'Stage 1 sleep|1', 
            'Stage 2 sleep|2', 
            1, 
            2, 
            'stage1', # hsp I0004
            'stage2', # hsp I0004
        ),
        SLEEP_STAGE_DEEP_SLEEP: (
            'Deep Sleep',
            'Stage_3',
            'N3',
            'Stage_4',
            'N4',
            'Stage 3 sleep|3', 
            'Stage 4 sleep|4', 
            3,
            4,
            'stage3', # hsp I0004
        ),
        SLEEP_STAGE_REM_SLEEP: (
            'REM',
            'REM sleep|5', 
            5,
            'rem', # hsp I0004
        ),
    },

    # Respiratory Groupings
    'Respiratory Events': {
        RESPIRATORY_EVENT_CENTRAL_APNEA: (
            'Central Apnea',

            'Central apnea|Central Apnea',
            'CENTRAL APNEA',
            'Apnea',
            'Central Apnea',
            'centralapnea', # hsp I0004
            
        ),
        RESPIRATORY_EVENT_OBSTRUCTIVE_APNEA: (
            'Obstructive Apnea',
            'Obstructive apnea|Obstructive Apnea',
            'OBSTRUCTIVE APNEA',
            'OBS Apnea',
            'Obst Apnea',
            'Obst. Apnea',
            'Obstructive Apnea',
            'obstructiveapnea', # hsp I0004
        ),
        RESPIRATORY_EVENT_MIXED_APNEA: (
            'Mixed Apnea',
            'Mixed apnea|Mixed Apnea',
            'MIXED APNEA',
            'Mixed Apnea',
            'Mixed Apnea',
            'mixedapnea', # hsp I0004
            
        ),
        RESPIRATORY_EVENT_HYPOPNEA: (
            'Hypopnea',
            'Hypopnea|Hypopnea',
            'HYPOPNEA',
            'Central Hypopnea',
            'Hypopnea',
            'Hypopnea',
            'hypopnea', # hsp I0004
            
        ),
        RESPIRATORY_EVENT_DESATURATION: (
            'Oxygen Desaturation',
            'Obstructive apnea|Obstructive Apnea',
            'DESATURATION',
            'SpO2 desaturation|SpO2 desaturation', 
            'SaO2',
            'Oxygen Desaturation',
            'desaturation w/ respiratory', # hsp I0004
            'desaturation', # hsp I0004
        ),
    },

    # Limb Movement Groupings
    # NOTE: for hsp I0004, periodic leg movement events were named 'right leg', 'both leg', 'left leg'
        # it is possible that other datasets use similar naming convention for ISOLATED leg movement events 
        # to avoid any bugs for other datasets, I decided to hard-code the event name convergions in the hsp_dataset.py file
    'Limb Movements': {
        LIMB_MOVEMENT_ISOLATED: (

            'LM-ISOLATED',

        ),
        LIMB_MOVEMENT_PERIODIC: (
        ),
        LIMB_MOVEMENT_ISOLATED_LEFT: (
            'Leg movement - left', # it's a nsrr annotation label
            'Limb Movement (Left)',
            'Limb movement - left|Limb Movement (Left)',
        ),
        LIMB_MOVEMENT_ISOLATED_RIGHT: (
            'Leg movement - right', # it's a nsrr annotation label
            'Limb Movement (Right)',
            'Limb movement - right|Limb Movement (Right)',
        ),
        LIMB_MOVEMENT_PERIODIC_LEFT: (
            'Periodic leg movement - left',
            'PLM (Left)',
            'Periodic leg movement - left|PLM (Left)',
        ),
        LIMB_MOVEMENT_PERIODIC_RIGHT: (
            'Periodic leg movement - right',
            'PLM (Right)',
            'Periodic leg movement - right|PLM (Right)',
        ),
    },


    # Arousals Groupings
    'Arousal Events': {
        AROUSAL_EVENT_CLASSIC: (
            'Arousal', # it's a nsrr annotation label
            'Spontaneous Arousal',
            'Arousal|Arousal ()',
            'Spontaneous arousal|Arousal (ARO SPONT)',
            'ASDA arousal|Arousal (ASDA)',
            'AROUSAL-SPONTANEOUS',
            'AROUSAL-NOISE',
            'Arousal|Arousal (Standard)',
            'Arousal|Arousal (STANDARD)',
            'Arousal|Arousal (Asda)',
            'arousal', # hsp I0004

        ),
        AROUSAL_EVENT_RESPIRATORY: (
            'RERA',
            'Respiratory Effort-Related Arousal',
            'AROUSAL-RESPIRATORY EVENT',
            'AROUSAL-SNORE',
            'Arousal resulting from respiratory effort|Arousal (ARO RES)',
            'Respiratory effort related arousal|RERA',
            'rera', # hsp I0004
            'arousal w/ respiratory', # hsp I0004

        ),
        AROUSAL_EVENT_EMG: (
            'EMG-Related Arousal',
            'Limb Arousal',

            'AROUSAL-LM',
            'AROUSAL-PLM',
            'Arousal resulting from Chin EMG|Arousal (CHESHIRE)',
            'Arousal resulting from Chin EMG|Arousal (Cheshire)',
            'Arousal|Arousal (ARO Limb)',

            'arousal w/ plm', # hsp I0004
            'both leg w/ arousal', # hsp I0004
            'right leg w/ arousal', # hsp I0004
            'left leg w/ arousal', # hsp I0004

        ),
    },

    # Lights on/off Groupings
    # --- this might only be relevant for hsp => we can use this to extract lights on/off events from text-based annotations
    # --- these are hand-curated based on a sample of 600 files from S0001 in HSP. I will expand as I onboard other sites
    'Lights Events': {
        LIGHTS_ON: (
            'Lights On',
            'LIGHTS ON', 
            'Lights_On', 
            'lights on', 
            'Lights On', 
            'LIGHTS /ON',
        ),
        LIGHTS_OFF: (
            'Lights Off',
            'LIGHTS OUT', 
            'Lights_Off', 
            'lights out', 
            'Lights_Off1', 
            'LIGHTS OUT ', 
            'Lights Out', 
            'Lights out', 
            'LIGHTS/OUT',
        ),
    },

}

SLEEP_STAGE_INVERSE_MAPPING = {v: k for k, values in EVENT_NAME_MAPPING['Sleep Stages'].items() for v in values}
RESPIRATORY_INVERSE_MAPPING = {v: k for k, values in EVENT_NAME_MAPPING['Respiratory Events'].items() for v in values}
LIMB_MOVEMENT_INVERSE_MAPPING = {v: k for k, values in EVENT_NAME_MAPPING['Limb Movements'].items() for v in values}
AROUSAL_INVERSE_MAPPING = {v: k for k, values in EVENT_NAME_MAPPING['Arousal Events'].items() for v in values}
LIGHTS_INVERSE_MAPPING = {v: k for k, values in EVENT_NAME_MAPPING['Lights Events'].items() for v in values}


"""
This is the mapping used for body position. For most NSRR datasets body position is a standalone channel in the edf file, but it is vectorized into discrete integers. 
This mapping dictionary can be used to convert the integers to the corresponding body position. 

For HSP, body position is a text-based annotation in the annotations file. This mapping dictionary can be used to convert the text-based annotations to the corresponding body position. 
"""

# body position event labels as discrete integers, careful if mapping from vectorized integers to these labels (just make sure each row does not get remapped more than once)
BODY_POSITION_RIGHT = 0 # 'Right'
BODY_POSITION_LEFT = 1 # 'Left'
BODY_POSITION_SUPINE = 2 # 'Supine'
BODY_POSITION_PRONE = 3 # 'Prone'
BODY_POSITION_UPRIGHT = 4 # 'Upright'
BODY_POSITION_OTHER_UNKNOWN = -1 # 'Other/Unknown' # use this to fill if missing

POSITION_MAPPING = { 
    SHHS: {
        BODY_POSITION_RIGHT: (0,),
        BODY_POSITION_LEFT: (1,),
        BODY_POSITION_SUPINE: (2,),
        BODY_POSITION_PRONE: (3,),
        BODY_POSITION_UPRIGHT: (NA_FLAG,), # not available
        BODY_POSITION_OTHER_UNKNOWN: (NA_FLAG,),
    },
    MESA: {
        BODY_POSITION_RIGHT: (0,),
        BODY_POSITION_LEFT: (2,), 
        BODY_POSITION_SUPINE: (1,),
        BODY_POSITION_PRONE: (3,),
        BODY_POSITION_UPRIGHT: (4,),
        BODY_POSITION_OTHER_UNKNOWN: (NA_FLAG,),
    },
    HSP: {
        BODY_POSITION_RIGHT: (
            'Position - Right',
            'Position - Right - 1',
            'Body_Position:_Right',
            ),
        BODY_POSITION_LEFT: (
            'Position - Left',
            'Position - Left - 1',
            'Body_Position:_Left',
        ),
        BODY_POSITION_SUPINE: (
            'Position - Supine',
            'Position - Supine - 1',
            'Body_Position:_Supine', 
        ),
        BODY_POSITION_PRONE: (
            'Position - Prone',
            'Position - Prone - 1',
            'Body_Position:_Prone', 
        ),
        BODY_POSITION_UPRIGHT: (
            'Position - Sitting',
            'Position - Sitting - 1',
            'Body_Position:_Upright', 
        ),
        BODY_POSITION_OTHER_UNKNOWN: (
            'Position - Disconnect',
            'Position - Disconnect - 1',
        ),
    },
}



