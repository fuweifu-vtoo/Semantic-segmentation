from __future__ import print_function
import os
import torch
from torch.autograd import Variable
from PIL import Image
import numpy as np
from models.u_net import UNet
from models.seg_net import Segnet
from data_loader.dataset import input_transform, colorize_mask

#model = Segnet(3,2)
#model_path = './checkpoint/Segnet/model/netG_final.pth'

model = UNet(3, 2)
model_path = './checkpoint/Unet/model/netG_1.pth'
model.load_state_dict(torch.load(model_path, map_location='cpu'))
test_image_path = './data/train/src/21.png'
test_image = Image.open(test_image_path).convert('RGB')
print('Operating...')
img = input_transform(test_image)
img = img.unsqueeze(0)
img = Variable(img)
pred_image = model(img)
predictions = pred_image.data.max(1)[1].squeeze_(1).cpu().numpy()
prediction = predictions[0]
predictions_color = colorize_mask(prediction)
predictions_color.show()
