from __future__ import print_function
import os
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
from models.u_net import UNet
from models.seg_net import Segnet
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

class Transformer(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img_):
        img_ = img_.resize(self.size, self.interpolation)
        img_ = self.toTensor(img_)  
        img_.sub_(0.5).div_(0.5)   
        return img_

#model = Segnet(3,3)
#model_path = './checkpoint/Segnet/model/netG_final.pth'
model = UNet(3,3)
model_path = './checkpoint/Unet/model/netG_final.pth'
model.load_state_dict(torch.load(model_path,map_location='cpu'))

test_image_path = 'C:/Users/1/Desktop/11.png'
test_image = Image.open(test_image_path).convert('RGB')
print('Operating...')
transformer = Transformer((256, 256))
img = transformer(test_image)
img = img.unsqueeze(0)
img = Variable(img)
label_image = model(img)
label_image = label_image.squeeze(0)
show = ToPILImage()
a = show((label_image +1) /2)   ##转换的时候，会自动从0-1转换成0-256，所以0.5会变成127
print(a.getpixel((100,100)))
print(a.size)
a.show()
