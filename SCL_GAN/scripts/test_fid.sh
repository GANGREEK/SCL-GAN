set -ex
python test_fid.py \
--dataroot ./datasets/WHU \
--checkpoints_dir ./checkpoints \
--name WHU2 \
--gpu_ids 0 \
--model sc \
--num_test 0 \
--dataset_mode aligned \
--direction BtoA \

