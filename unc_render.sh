POINTS=$1
UNC_RESULT_DIR=results/$POINTS
mkdir -p $UNC_RESULT_DIR


python -u new.py \
    --log_dir log \
    --seed 42 \
    --test_batch_size 24 \
    --run_name unc_inference_$POINTS \
    --unc_result_dir $UNC_RESULT_DIR \
    --unc_round 50 \
    --unc_dataset ScannetVoxelization2cmtestDataset \
    --scannet_path /home/cloudroot/data/scannet_processed/$POINTS/train \
    --scannet_test_path /home/cloudroot/data/scannet_processed/full/train \
    --do_unc_render
