set -ex
python test.py \
--dataroot ./datasets/facades
--checkpoints_dir ./checkpoints --name facades \
--model sc \
--num_test 0
