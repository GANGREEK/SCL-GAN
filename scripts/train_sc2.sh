set -ex
python train.py  \
--dataroot ./datasets/WHU \
--name WHU2 \
--model sc \
--gpu_ids 0 \
--lambda_spatial 10 \
--lambda_gradient 0 \
--attn_layers 4,7,9 \
--loss_mode cos \
--gan_mode lsgan \
--direction BtoA \
--patch_size 64 \
--dataset_mode aligned \
--display_id 0 \
#\--learned_attn \--augment
