set -ex
python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --netG resnet_9blocks --direction BtoA --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0
