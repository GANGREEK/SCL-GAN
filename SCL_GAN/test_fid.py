"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import math
import numpy
import scipy.signal
import scipy.ndimage
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable, grad
#from torchsummary import summary
from PIL import Image 
import numpy as np 
import os
from collections import OrderedDict
import cv2
import tensorflow as tf
import numpy
import sklearn as sklearn
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import torchvision.models as models
from torch.nn import Module, Parameter, Softmax
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie1976
from skimage import io, color
import re
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from evaluations.fid_score import calculate_fid_given_paths
from util.visualizer import save_images
from util import html
import util.util as util
import matplotlib.pyplot as plt
from skimage import measure

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.num_test = opt.num_test if opt.num_test > 0 else float("inf")
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    # traverse all epoch for the evaluation
    files_list = os.listdir(opt.checkpoints_dir + '/' + opt.name)
    epoches = []
    fid_values = {}
    for file in files_list:
        if 'net_G' in file and 'latest' not in file:
            name = file.split('_')
            epoches.append(name[0])
    for epoch in epoches:
        opt.epoch = epoch
        model = create_model(opt)      # create a model given opt.model and other options
        # create a website
        web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
        print('creating web directory', web_dir)
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
        # test with eval mode. This only affects layers like batchnorm and dropout.
        # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
        # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
        for i, data in enumerate(dataset):
            if i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)  # regular setup: load and print networks; create schedulers
                model.parallelize()
                if opt.eval:
                    model.eval()
            if i >= opt.num_test:  # only apply our model to opt.num_test images.
                break
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()     # get image paths
            if i % 5 == 0:  # save images to an HTML file
                print('processing (%04d)-th image... %s' % (i, img_path))
            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        paths = [os.path.join(web_dir, 'images', 'fake_B'), os.path.join(web_dir, 'images', 'real_B')]
        
            
        dir_src = paths[1]
        dir_dst = paths[0]
        m_avg=0
        p_avg=0
        s_avg=0
        f_avg=0
        c_avg=0
        fs_avg=0
        n=0
        ssi = []
        def compare_images(imageA, imageB, title):
    
    
            s= measure.compare_ssim(imageA, imageB)	 
            print(s)
            
            #####################################
            
 
           
            # show the images
            #plt.show()
            return s
        

        for dirpath, dirs, files in os.walk(paths[0]):
            for filename in files:
                #print(filename)
                n=n+1
                #print filename # testing
                #filename1=filename.replace('real','fake')
                #print filename1
                dirpath1=dirpath.replace('real_B','fake_B')
                fname = os.path.join(paths[0], filename)
                fname1=os.path.join(paths[1],filename)
                org = cv2.imread(fname)
                #print(org.shape)
                gen = cv2.imread(fname1)
         
                lab1 = cv2.cvtColor(org,cv2.COLOR_BGR2LAB)
                lab2 = cv2.cvtColor(gen,cv2.COLOR_BGR2LAB)
                #res = color_loss(lab1,lab2)/(255*255)
        
                original = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
                generated = cv2.cvtColor(gen, cv2.COLOR_BGR2GRAY)
                org1 = cv2.cvtColor(org, cv2.COLOR_BGR2RGB)
                gen1 = cv2.cvtColor(gen, cv2.COLOR_BGR2RGB)
        
                images = ("Original", org1), ("Generated", gen1)

        
                ms=compare_images(original, generated, "Original vs. Generated")
               
               
               
                s_avg += ms
               
                
                
        ssi.append(s_avg/n)

        print ("SSIM: %.4f" % ( s_avg/n))
        
        fid_value = calculate_fid_given_paths(paths, 50, True, 2048)
        print (fid_value)
        f = open("result.txt", "a")
        f.write("%s %s %s\n" % ((s_avg/n), fid_value, epoch))
        fid_values[int(epoch)] = fid_value
        webpage.save()  # save the HTML
    
    x = []
    y = []
    for key in sorted(fid_values.keys()):
        x.append(key)
        y.append(fid_values[key])
    plt.figure()
    plt.plot(x, y)
    for a, b in zip(x, y):
        plt.text(a, b, str(round(b, 2)))
    plt.xlabel('Epoch')
    plt.ylabel('FID on test set')
    plt.title(opt.name)
    ax2=plt.twinx()
    ax2.plot(ssi, y, color="blue",marker="o")
    ax2.set_ylabel("SSIM",color="blue",fontsize=6)
    plt.show()
    
    
    plt.savefig(os.path.join(opt.results_dir, opt.name, 'fid.jpg'))

class Vgg19Face(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19Face, self).__init__()
        vgg_pretrained_features = models.vgg19()
        ####################################################################print(vgg_pretrained_features)
        vgg_pretrained_features.eval()
        checkpoint = torch.load('./models/VGG/VGG19_FACE_checkpoint.pth.tar')
        state_dict =checkpoint['state_dict']
        #print(state_dict)
        new_state_dict = OrderedDict()
        index_layer=0
        for k, v in state_dict.items():
            if index_layer<=31:
                name = k[0:9]+k[16:] # remove `module.`
                new_state_dict[name] = v
            else:
                new_state_dict[k]=v
            index_layer=index_layer+1
        vgg_pretrained_features.load_state_dict(new_state_dict)
        vgg_pretrained_features=vgg_pretrained_features.features
        vgg_pretrained_features=vgg_pretrained_features.cuda()

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGFaceLoss(nn.Module):
    def __init__(self):
        super(VGGFaceLoss, self).__init__()        
        self.vgg = Vgg19Face().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        #loss+=sklearn.metrics.pairwise.cosine_similarity(x_vgg, y_vgg)
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss
def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err=np.mean((imageA-imageB)**2)

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def vifp_mscale(ref, dist):
    sigma_nsq=2
    eps = 1e-10

    num = 0.0
    den = 0.0
    for scale in range(1, 5):
       
        N = 2**(4-scale+1) + 1
        sd = N/5.0

        if (scale > 1):
            ref = scipy.ndimage.gaussian_filter(ref, sd)
            dist = scipy.ndimage.gaussian_filter(dist, sd)
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]
                
        mu1 = scipy.ndimage.gaussian_filter(ref, sd)
        mu2 = scipy.ndimage.gaussian_filter(dist, sd)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = scipy.ndimage.gaussian_filter(ref * ref, sd) - mu1_sq
        sigma2_sq = scipy.ndimage.gaussian_filter(dist * dist, sd) - mu2_sq
        sigma12 = scipy.ndimage.gaussian_filter(ref * dist, sd) - mu1_mu2
        
        sigma1_sq[sigma1_sq<0] = 0
        sigma2_sq[sigma2_sq<0] = 0
        
        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12
        
        g[sigma1_sq<eps] = 0
        sv_sq[sigma1_sq<eps] = sigma2_sq[sigma1_sq<eps]
        sigma1_sq[sigma1_sq<eps] = 0
        
        g[sigma2_sq<eps] = 0
        sv_sq[sigma2_sq<eps] = 0
        
        sv_sq[g<0] = sigma2_sq[g<0]
        g[g<0] = 0
        sv_sq[sv_sq<=eps] = eps
        
        num += numpy.sum(numpy.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += numpy.sum(numpy.log10(1 + sigma1_sq / sigma_nsq))
        
    vifp = num/den

    if numpy.isnan(vifp):
        return 1.0
    else:
        return vifp
def color_loss(lab1, lab2):
    
    res=0;
   
    width =lab1.shape[0]
    
   
    #lab1 = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)

    L1,A1,B1=cv2.split(lab1)
    #img2 = image2
    #lab2 = cv2.cvtColor(img2,cv2.COLOR_BGR2LAB)
    L2,A2,B2=cv2.split(lab2)
    for i in range(width):
        for k in range(width):
            color1 = LabColor(lab_l=L1[i][k], lab_a=A1[i][k], lab_b=B1[i][k])
# Color to be compared to the reference.
            color2 = LabColor(lab_l=L2[i][k], lab_a=A2[i][k], lab_b=B2[i][k])
# This is your delta E value as a float.
            delta_e = delta_e_cie1976(color1, color2)
            res+= delta_e
            #print(delta_e)
    return res

def compare2(ImageAPath, ImageBPath):
    
    img1 = cv2.imread(ImageAPath)          # queryImage
    img2 = cv2.imread(ImageBPath)
    img11 = np.expand_dims(img1, axis=3)
    img12 = np.expand_dims(img2, axis=3)
    image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    A1=Image.open(ImageAPath)# trainImage
    B1=Image.open(ImageBPath)
    A =  transforms.ToTensor()(A1).unsqueeze_(0).to('cuda')
    B =  transforms.ToTensor()(B1).unsqueeze_(0).to('cuda')
    #Vgg=Vgg19Face()
    #AL=Vgg19Face(A)
    #print(AL)
    vggloss= VGGFaceLoss()
    m = vggloss(A,B)
    print(m)
    
    
    faceloss = m
    
   
    return faceloss







