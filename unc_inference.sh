POINTS=$1
UNC_RESULT_DIR=results/$POINTS
mkdir -p $UNC_RESULT_DIR


python -u new.py \
    --log_dir log \
    --seed 42 \
    --unc_dataset ScannetVoxelization2cmtestDataset \
    --scannet_path /home/cloudroot/data/scannet_processed/$POINTS/train \
    --scannet_test_path /home/cloudroot/data/scannet_processed/full/train \
    --model Res16UNet34CUNC \
    --weights /mnt/air-02/luoly/tbw/log/checkpoint_NoneRes16UNet34C_20000.pth \
    --checkpoint_dir checkpoints \
    --num_workers 0 \
    --test_batch_size 24 \
    --run_name unc_inference_$POINTS \
    --resume /home/cloudroot/3DScanSeg/checkpoints/finetune_b_${POINTS}_best_bak.pth \
    --unc_result_dir $UNC_RESULT_DIR \
    --unc_round 50 \
    --do_unc_inference
