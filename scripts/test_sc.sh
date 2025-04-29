set -ex
python test.py \
--dataroot ./datasets/WHU 
--checkpoints_dir ./checkpoints --name WHU22 \
--model sc \

