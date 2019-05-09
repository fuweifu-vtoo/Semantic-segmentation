import numpy as np
import torch
import torch.utils.data as data
import os
from PIL import Image
import random


def is_image_file(filename):  # 定义一个判断是否是图片的函数
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def trans_to_tensor(pic):  # 定义一个转变图像格式的函数
    if isinstance(pic, np.ndarray):
        img = torch.from_numpy(pic.transpose((2, 0, 1)))  # transpose和reshape区别巨大
        return img.float().div(255)
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img
"""
def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


def rotate(xb, yb, angle):
    M_rotate = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb, yb


def blur(img):
    img = cv2.blur(img, (3, 3));
    return img


def add_noise(img):
    for i in range(200):  # 添加点噪声
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    return img
"""

def data_augment(img1, img2, flip=1, ROTATE_90=1, ROTATE_180=1, ROTATE_270=1, add_noise=1):
    n = flip + ROTATE_90 + ROTATE_180 + ROTATE_270 + add_noise
    a = random.random()
    if flip == 1:
        img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
    if ROTATE_90 == 1:
        img1 = img1.transpose(Image.ROTATE_90)
        img2 = img2.transpose(Image.ROTATE_90)
    if ROTATE_180 == 1:
        img1 = img1.transpose(Image.ROTATE_180)
        img2 = img2.transpose(Image.ROTATE_180)
    if ROTATE_270 == 1:
        img1 = img1.transpose(Image.ROTATE_270)
        img2 = img2.transpose(Image.ROTATE_270)
    if add_noise == 1:
        pass

class train_dataset(data.Dataset):
    def __init__(self, data_path='', size_w=256, size_h=256, flip=0):
        super(train_dataset, self).__init__()
        self.list = [x for x in os.listdir(data_path + '/src/') if is_image_file(x)]
        self.data_path = data_path
        self.size_w = size_w
        self.size_h = size_h
        self.flip = flip

    def __getitem__(self, index):
        initial_path = os.path.join(self.data_path + '/src/', self.list[index])
        semantic_path = os.path.join(self.data_path + '/label/', self.list[index])
        assert os.path.exists(semantic_path)
        try:
            initial_image = Image.open(initial_path).convert('RGB')
            semantic_image = Image.open(semantic_path).point(lambda i: i * 80).convert('RGB')
        except OSError:
            return None, None, None

        initial_image = initial_image.resize((self.size_w, self.size_h), Image.BILINEAR)
        semantic_image = semantic_image.resize((self.size_w, self.size_h), Image.BILINEAR)

        if self.flip == 1:
            a = random.random()
            if a < 1 / 3:
                initial_image = initial_image.transpose(Image.FLIP_LEFT_RIGHT)
                semantic_image = semantic_image.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                if a < 2 / 3:
                    initial_image = initial_image.transpose(Image.ROTATE_90)
                    semantic_image = semantic_image.transpose(Image.ROTATE_90)

        initial_image = trans_to_tensor(initial_image)
        initial_image = initial_image.mul_(2).add_(-1)  # -1到1之间
        semantic_image = trans_to_tensor(semantic_image)
        semantic_image = semantic_image.mul_(2).add_(-1)

        return initial_image, semantic_image, self.list[index]

    def __len__(self):
        return len(self.list)

# if __name__ == '__main__':
#     data = train_dataset(data_path='../data/train')
#     n = len(data.list)
#     for i in range(n):
#         data.__getitem__(i)
#     print(len(data.list))
