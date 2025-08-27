source activate myenv
module load anaconda3_gpu
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v "/sw/external/libraries/cudnn-linux-x86_64-8.9.0.131_cuda11-archive/lib" | paste -sd ':' -)
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
/u/ztshuai/.conda/envs/myenv/bin/python step1_large_patch_sleep_postprocess.py
#/u/ztshuai/.conda/envs/myenv/bin/python preprocess_main.py --folder /scratch/besp/shared_data/wsc --output-folder /scratch/besp/shared_data/test_new/preprocessed_wsc --max-parallel 1