#!/bin/bash

set -x
# Exit script when a command returns nonzero state
set -e

set -o pipefail
#export CUDA_VISIBLE_DEVICES=1,3
PYTHONUNBUFFERED="True"
DATAPATH=/data/hdd01/luoly/min_eff/eff_200/train
TESTDATAPATH=/data/hdd01/luoly/Minkowski/scan_processed/scan_processed/train
 # Download ScanNet segmentation dataset and change the path here
# PRETRAIN=/data/hdd01/luoly/tbw/log/512_finetune_20000_iter_200/checkpoint_NoneRes16UNet34C.pth # For finetuning, use the checkpoint path here.
PRETRAIN=/data/hdd01/luoly/tbw/log/checkpoint_NoneRes16UNet34C_20000.pth
MODEL=Res16UNet34C
BATCH_SIZE=${BATCH_SIZE:-8}
TIME=$(date +"%Y-%m-%d_%H-%M-%S")
# LOG_DIR=/home/aidrive1/workspace/luoly/dataset/Min_scan/eff_200/fps_log/200/512/tmp_dir_scannet
LOG_DIR=/data/hdd01/luoly/tbw/log/512_finetune_20000_iter_200
mkdir -p $LOG_DIR
LOG="$LOG_DIR/$TIME.txt"

python -m ddp_main \
    --distributed_world_size=1 \
    --train_phase=train \
    --is_train=True \
    --lenient_weight_loading=True \
    --stat_freq=1 \
    --val_freq=500 \
    --save_freq=500 \
    --model=${MODEL} \
    --conv1_kernel_size=5 \
    --normalize_color=True \
    --dataset=ScannetVoxelization2cmDataset \
    --testdataset=ScannetVoxelization2cmtestDataset \
    --batch_size=$BATCH_SIZE \
    --num_workers=1 \
    --num_val_workers=1 \
    --scannet_path=${DATAPATH} \
    --scannet_test_path=${TESTDATAPATH} \
    --return_transformation=False \
    --test_original_pointcloud=False \
    --save_prediction=False \
    --lr=1.631e-02 \
    --scheduler=PolyLR \
    --max_iter=4000 \
    --log_dir=${LOG_DIR} \
    --weights=${PRETRAIN} \
    $3 2>&1 | tee -a "$LOG"
