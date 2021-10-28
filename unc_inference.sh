POINTS=$1
UNC_RESULT_DIR=results/$POINTS
mkdir -p $UNC_RESULT_DIR


python -u new.py \
    --log_dir log \
    --seed 42 \
    --unc_dataset ScannetVoxelization2cmtestDataset \
    --scannet_path ~/data/$POINTS/train \
    --scannet_test_path ~/data/full/train \
    --model Res16UNet34CUNC \
    --checkpoint_dir checkpoints \
    --num_workers 2 \
    --test_batch_size 24 \
    --run_name unc_inference_$POINTS \
    --resume ~/3DScanSeg/checkpoints/finetune_b_${POINTS}_best.pth \
    --unc_result_dir $UNC_RESULT_DIR \
    --unc_round 50 \
    --do_unc_inference
