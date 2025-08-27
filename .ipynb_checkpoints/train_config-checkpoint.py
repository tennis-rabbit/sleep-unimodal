from wav2sleep.config import *
MODEL_LIST = ["clip", "simclr", "mae", "dino"]
SPLIT_DATA_FOLDER = "/projects/besp/shared_data/real_sleep_postprocessed_data/five_min_sleep_data_split_test_with_cache"
TRAIN_EDF_COLS = [ECG, HR, PPG,
            SPO2, OX, ABD, THX, AF, NP, SN, 
            EOG_E1_A2, EOG_E2_A1,
            EMG_LLeg, EMG_RLeg, EMG_Chin,
            EEG_C3_A2, EEG_C4_A1, EEG_F3_A2, EEG_F4_A1, EEG_O1_A2, EEG_O2_A1,
            POS,
            # EEG_F4_A1_multitaper_Alpha, EEG_F4_A1_multitaper_Beta, EEG_F4_A1_multitaper_Delta, EEG_F4_A1_multitaper_Gamma, EEG_F4_A1_multitaper_Theta,
           ] # for testing usage
CKPT_PATH = "/scratch/besp/shared_data/test_zitao_pretrain"
LOG_PATH = "/scratch/besp/shared_data/test_zitao_pretrain"