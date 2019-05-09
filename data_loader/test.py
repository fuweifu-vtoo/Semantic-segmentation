import numpy as np
import torch
import torch.utils.data as data
import os
from PIL import Image
import random

semantic_image = Image.open('../data/train/label/0.png').point(lambda i: i * 80).convert('RGB')
w,h = semantic_image.size
for row in range(w):
	for cloumn in range(h):
			print(semantic_image.getpixel((row,cloumn)))
semantic_image.show()