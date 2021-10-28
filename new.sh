POINTS=$1
python -u new.py \
    --log_dir log \
    --seed 42 \
    --train_dataset ScannetVoxelization2cmDataset \
    --val_dataset ScannetVoxelization2cmtestDataset \
    --scannet_path /home/aidrive/tb5zhh/data/$POINTS/train \
    --scannet_test_path /home/aidrive/tb5zhh/data/full/train \
    --model Res16UNet34CUNC \
    --weights checkpoints/pretrain_20000.pth \
    --checkpoint_dir checkpoints \
    --num_workers 4 \
    --validate_step 50 \
    --optim_step 1 \
    --train_batch_size 16   \
    --val_batch_size 24  \
    --lr 0.1 \
    --save_epoch 5 \
    --max_iter 30000 \
    --run_name finetune_b_$POINTS \
    --scheduler PolyLR \
    --do_train \
    --resume /home/aidrive/tb5zhh/3DScanSeg/checkpoints/finetune_b_${POINTS}_latest.pth \
    --wandb True
    

# finetune_a: move dropout layer before the linear layer
# finetune_b: smaller batch size
