#!/bin/bash
#SBATCH --job-name 50_fit_unc # 作业名为 example
#SBATCH --output job_log/50_fit_unc_%J.out    # 屏幕上的输出文件重定向到 [JOBID].out
#SBATCH --gres gpu:a100:1  # 使用 1 张 A100 显卡
#SBATCH --requeue
#SBATCH --time 3-0
NAME=50_fit_unc

echo $NAME
DATASET_PATH=/home/aidrive/tb5zhh/data/${NAME}/train
TRAIN_BATCH_SIZE=16
LR=0.1
MODEL=Res16UNet34C
RUN_NAME=finetune_${NAME}_${LR}_${MODEL}
python -u new.py \
    --log_dir log \
    --seed 42 \
    --train_dataset ScannetVoxelization2cmDataset \
    --val_dataset ScannetVoxelization2cmtestDataset \
    --scannet_test_path /home/aidrive/tb5zhh/data/full/train \
    --checkpoint_dir checkpoints \
    --num_workers 4 \
    --validate_step 100 \
    --optim_step 1 \
    --val_batch_size 8  \
    --save_epoch 1 \
    --max_iter 30000 \
    --scheduler PolyLR \
    --do_validate \
    --weights /home/aidrive/tb5zhh/new_3dseg/stsegmentation/log/200_fit_sdouble/checkpoint_NoneRes16UNet34Cbest_val.pth\
    --run_name $RUN_NAME \
    --model $MODEL \
    --lr $LR \
    --train_batch_size $TRAIN_BATCH_SIZE  \
    --scannet_path $DATASET_PATH \
    --wandb False
    # --resume 1
    
# sleep 30
# finetune_a: move dropout layer before the linear layer
# finetune_b: smaller batch size

# --log_dir log --seed 42 --train_dataset ScannetVoxelization2cmDataset --val_dataset ScannetVoxelization2cmtestDataset --scannet_test_path /home/aidrive/tb5zhh/data/full/train --checkpoint_dir checkpoints --num_workers 4 --validate_step 50 --optim_step 1 --val_batch_size 24 --save_epoch 5 --max_iter 30000 --scheduler PolyLR --do_train --run_name finetune_double_mixture_nounc_200 --weights checkpoints/pretrain_20000.pth --model Res16UNet34C --lr 0.1 --train_batch_size 16 --scannet_path /home/aidrive/tb5zhh/data/200_fit_double/train --wandb True