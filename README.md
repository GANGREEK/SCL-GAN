# Spatially-Correlative Lightweight GAN For Thermal to Visible Face Transformation
to Visible Face Transformation



We provide the Pytorch implementation of "Spatially-Correlative Lightweight GAN for Thermal
to Visible Face Transformation". Based on the inherent self-similarity of facial attributes.

## ToDo
- a simple example 

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
https://github.com/GANGREEK/SCL-GAN.git
cd 
```


An sample Code For SCL-GAN paper

Required Library
For the Jetson Deployment Necessary Files and Steps: 

Step 1: Configure the Jeston-TX2 Board by using Jetpack over the SDK manager.

Step 2: Install PyTorch for the Jetson Borad by using the link https://github.com/Qengineering/PyTorch-Jetson-Nano.

Step 3: Install the packages :
‘argparse==1.2.1’, ‘attrs==17.4.0’, ‘funcsigs==1.0.2’, ‘gps==3.17’,‘graphsurgeon==0.4.5’, ‘jetson.gpio==2.0.8’, ‘numpy==1.13.3’,‘pluggy==0.6.0’, ‘pycairo==1.16.2’, ‘pygobject==3.26.1’, ‘py==1.5.2’,‘pytest==3.3.2’, ‘python==2.7.17’, ‘six==1.11.0’, \\ ‘tensorrt==7.1.3.0’, ‘uff==0.6.9’,‘unity-lens-photos==1.0’, ‘urwid==2.0.1’, ‘wsgiref==0.1.2’.

Step 4: For visualizing of the generated images over the screen attached to the board install: Matplot and NumPy from https://forums.developer.nvidia.com/t/jetson-nano-how-can-install-matplotlib.

