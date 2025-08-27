       
source activate melp
module load anaconda3_gpu
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v "/sw/external/libraries/cudnn-linux-x86_64-8.9.0.131_cuda11-archive/lib" | paste -sd ':' -)
export SLURM_JOB_NAME=bash
export WANDB_API_KEY 09d8990dc4772f6f248de810bb3403b810d78553
# CUDA_VISIBLE_DEVICES=0 /u/ztshuai/.conda/envs/melp/bin/python main_pretrain.py --num_workers 8 --num_devices 1 --num_nodes 1\
#     --lr 2e-4 --model_name "clip" --batch_size 40 --max_epochs 100 \
#     --clip_loss_weight 1.0 --caption_loss_weight 2.0 --local_loss_weight 0.2
    
    
CUDA_VISIBLE_DEVICES=0,1,2,3 /u/ztshuai/.conda/envs/melp/bin/python main_pretrain.py\
    --num_workers 8 --num_devices 4 --num_nodes 1\
    --lr 2e-4 --batch_size 40 --max_epochs 100 \
    --model_name "dino" --psg_encoder_name 'vit_small'\
    --clip_loss_weight 1.0\
    --wandb_proj_name "dino_pretrain"\
