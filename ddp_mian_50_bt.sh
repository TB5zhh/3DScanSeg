#!/bin/bash

set -x
# Exit script when a command returns nonzero state
set -e

set -o pipefail
export CUDA_VISIBLE_DEVICES=5,8
BATCH_SIZE=${BATCH_SIZE:-8}
DATAPATH=/data/hdd01/luoly/min_eff/eff_50/train
TESTDATAPATH=/data/hdd01/luoly/Minkowski/scan_processed/scan_processed/train
LOG_DIR=/data/hdd01/luoly/tbw/log/512_finetune_20000_iter_50
PRETRAIN=/data/hdd01/luoly/tbw/log/checkpoint_NoneRes16UNet34C_20000.pth # For finetuning, use the checkpoint path here.

MODEL=Res16UNet34C
TIME=$(date +"%Y-%m-%d_%H-%M-%S")
mkdir -p $LOG_DIR
LOG="$LOG_DIR/$TIME.txt"
PYTHONUNBUFFERED="True"

python -m ddp_main \
    --distributed_world_size=2 \
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
    --lr=0.1 \
    --scheduler=PolyLR \
    --max_iter=30000 \
    --log_dir=${LOG_DIR} \
    --weights=${PRETRAIN} \
    $3 2>&1 | tee -a "$LOG"
