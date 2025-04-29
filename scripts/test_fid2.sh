set -ex
python test_fid.py \
--dataroot ./datasets/CHILD \
--checkpoints_dir ./checkpoints \
--name CHILDM \
--gpu_ids 0 \
--model sc \
--num_test 0 \
--dataset_mode aligned \
--direction BtoA \

