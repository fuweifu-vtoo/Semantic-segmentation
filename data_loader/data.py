# import cv2
import os
from PIL import Image
import numpy as np
import torch as t

#
root = '.'
root = '../checkpoint/training_results'
path = os.path.join(root, 'fake_samples_epoch_001_1600.png')
# img = Image.open(path).convert('L')
img = Image.open(path)
out = img.point(lambda i: i * 80)
out1 = out.convert('RGB')
Image._show(img)
Image._show(out1)

# img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# print(img.shape)
# img1 = t.Tensor(img)
# img2 = img1.mul_(20)
# print(img1.shape)
# img3 = img2.numpy()
# # cv2.imshow('img3', img3)
# # cv2.waitKey()
# # a = np.ones([256, 256, 3], dtype=np.int8)
# cv2.imshow('1', img)
# cv2.waitKey()
# # cv2.imshow('2', a)
# # cv2.waitKey()
# # cv2.imshow('test', a)
# # cv2.waitKey()
