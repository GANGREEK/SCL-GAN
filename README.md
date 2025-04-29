
# Spatially-Correlative Lightweight GAN For Thermal to Visible Face Transformation



We provide the Pytorch implementation of "Spatially-Correlative Lightweight GAN for Thermal
to Visible Face Transformation". Based on the inherent self-similarity of facial attributes.

 

## Getting Started

### Installation
This code was tested with Pytorch 1.7.0, CUDA 10.2, and Python 3.7

- Install Pytoch 1.7.0, torchvision, and other dependencies from [http://pytorch.org](http://pytorch.org)
- Install python libraries [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate) for visualization

```
pip install visdom dominate
```
- Clone this repo:

```
git clone https://github.com/GANGREEK/SCL-GAN.git
cd SCL-GAN

```


### Training



```
sh ./scripts/train_sc.sh 
```

- Set ```--use_norm``` for cosine similarity map, the default similarity is dot-based attention score. ```--learned_attn, --augment``` for the learned self-similarity.
- To view training results and loss plots, run ```python -m visdom.server``` and copy the URL [http://localhost:port](http://localhost:port).
- Training models will be saved under the **checkpoints** folder.
- The more training options can be found in the **options** folder.
<br><br>


- Train the *single-image* translation model:

```
sh ./scripts/train_sinsc.sh 
```

### Testing


sh ./scripts/test_sc.sh
```

- Test the *single-image* translation model:

```
sh ./scripts/test_sinsc.sh
```

- Test the FID score for all training epochs:

```
sh ./scripts/test_fid.sh
```

### Pretrained Models

Download the pre-trained models (will be released soon) using the following links and put them under```checkpoints/``` directory.

)
```
## Citation
Submitted in In ICIP-2025(Will update Soon after good news)

## Acknowledge
Our code is developed based on [CUT](https://github.com/taesungp/contrastive-unpaired-translation) and [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). We also thank [pytorch-fid](https://github.com/mseitzer/pytorch-fid) for FID computation,  [LPIPS](https://github.com/richzhang/PerceptualSimilarity) for diversity score.



