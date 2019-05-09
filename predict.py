from __future__ import print_function
import os
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
# from models.u_net import UNet
from models.seg_net import Segnet
import torchvision.transforms as transforms

class Transformer(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img_):
        img_ = img_.resize(self.size, self.interpolation)
        img_ = self.toTensor(img_)   ##这部分好像会把（0,255）转到（0,1）
        img_.sub_(0.5).div_(0.5)   ##这部分将范围从（0,1）变成（-1,1），和模型自身的输入要匹配
        return img_


model = Segnet(3,3)
model_path = './checkpoint/training_results_segnet/model/netG_final.pth'
model.load_state_dict(torch.load(model_path,map_location='cpu'))


test_image_path = './data/train/src/0.png'
test_image = Image.open(test_image_path).convert('RGB')
print('Operating...')
transformer = Transformer((256, 256))
img = transformer(test_image)
img = img.unsqueeze(0)
img = Variable(img)
label_image = model(img)


label_image_save_path = './data/train/label'
vutils.save_image(label_image.data.reshape(-1,3,256,256), label_image_save_path + '/0_label_image.png',normalize=True)