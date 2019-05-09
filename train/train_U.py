from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torch.autograd import Variable
import time

from data_loader.dataset import train_dataset
from models.u_net import UNet
from models.seg_net import Segnet
import MIoU

parser = argparse.ArgumentParser(description='Training a UNet model')
parser.add_argument('--batch_size', type=int, default=16, help='equivalent to instance normalization with batch_size=1')
parser.add_argument('--input_nc', type=int, default=3)
parser.add_argument('--output_nc', type=int, default=3)
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--manual_seed', type=int, help='manual seed')
parser.add_argument('--num_workers', type=int, default=6, help='how many threads of cpu to use while loading data')
parser.add_argument('--size_w', type=int, default=256, help='scale image to this size')
parser.add_argument('--size_h', type=int, default=256, help='scale image to this size')
parser.add_argument('--flip', type=int, default=0, help='1 for flipping image randomly, 0 for not')
parser.add_argument('--net', type=str, default='', help='path to pre-trained network')
parser.add_argument('--data_path', default='', help='path to training images')
parser.add_argument('--outf', default='', help='folder to output images and model checkpoints')
parser.add_argument('--save_epoch', default=5, help='path to val images')
parser.add_argument('--test_step', default=300, help='path to val images')
parser.add_argument('--log_step', default=1, help='path to val images')
parser.add_argument('--num_GPU', default=2, help='number of GPU')
opt = parser.parse_args()
opt.cuda = True
opt.data_path = '../data/train'
opt.outf = '../checkpoint/training_results_unet'
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

try:
    os.makedirs(opt.outf + '/model/')
except OSError:
    pass

if opt.manual_seed is None:
    opt.manual_seed = random.randint(1, 10000)
print("Random Seed: ", opt.manual_seed)
random.seed(opt.manual_seed)
torch.manual_seed(opt.manual_seed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manual_seed)

cudnn.benchmark = True

###########   DATASET   ###########
train_datatset_ = train_dataset(opt.data_path, opt.size_w, opt.size_h, opt.flip)
train_loader = torch.utils.data.DataLoader(dataset=train_datatset_, batch_size=opt.batch_size, shuffle=True,
                                           num_workers=opt.num_workers)


###########   MODEL   ###########
def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


net = UNet(opt.input_nc, opt.output_nc)

if opt.net != '':
    net.load_state_dict(torch.load(opt.netG))
else:
    net.apply(weights_init)
if opt.cuda:
    net.cuda()
if opt.num_GPU > 1:
    net=nn.DaraParallel(net)

# print(net)

###########   LOSS & OPTIMIZER   ##########
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
###########   GLOBAL VARIABLES   ###########
initial_image = torch.FloatTensor(opt.batch_size, opt.input_nc, opt.size_w, opt.size_h)
semantic_image = torch.FloatTensor(opt.batch_size, opt.input_nc, opt.size_w, opt.size_h)

initial_image = Variable(initial_image)
semantic_image = Variable(semantic_image)

if opt.cuda:
    initial_image = initial_image.cuda()
    semantic_image = semantic_image.cuda()

if __name__ == '__main__':
    ########### Training   ###########
    log = open('log.txt', 'w')
    start = time.time()
    net.train()
    for epoch in range(1, opt.niter+1):
        loader = iter(train_loader)
        for i in range(0, train_datatset_.__len__(), opt.batch_size):
            initial_image_, semantic_image_, name = loader.next()

            initial_image.data.resize_(initial_image_.size()).copy_(initial_image_)
            semantic_image.data.resize_(semantic_image_.size()).copy_(semantic_image_)

            semantic_image_pred = net(initial_image)

            initial_image = initial_image.view(-1)     #拉伸成为一维tensor
            semantic_image_pred = semantic_image_pred.view(-1)

            loss = criterion(semantic_image_pred, semantic_image)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ########### Logging ##########
            if i % opt.log_step == 0:
                print('[%d/%d][%d/%d] Loss: %.4f' %
                      (epoch, opt.niter, i, len(train_loader) * opt.batch_size, loss.item()))
                log.write('[%d/%d][%d/%d] Loss: %.4f' %
                          (epoch, opt.niter, i, len(train_loader) * opt.batch_size, loss.item()))
            if i % opt.test_step == 0:
               vutils.save_image(semantic_image_pred.data.reshape(-1,3,256,256), opt.outf + '/fake_samples_epoch_%03d_%03d.png' % (epoch, i),normalize=True)

            

        # if epoch % opt.val_epoch == 0:
        #     loader_synth = iter(val_loader_synth)
        #     loader_real = iter(val_loader_real)
        #
        #     val_distorted_synth, name = loader_synth.next()
        #     val_distorted_real, name = loader_real.next()
        #
        #     real_A.data.resize_(val_distorted_synth.size()).copy_(val_distorted_synth)
        #     val_corrected_synth = netG(real_A)
        #     vutils.save_image(val_corrected_synth.data,
        #                       opt.outf + 'val_synth/fake_samples_epoch_%03d.png' % (epoch), normalize=True)
        #
        #     real_A.data.resize_(val_distorted_real.size()).copy_(val_distorted_real)
        #     val_corrected_real = netG(real_A)
        #     vutils.save_image(val_corrected_real.data,
        #                       opt.outf + 'val_real/fake_samples_epoch_%03d.png' % (epoch), normalize=True)

        if epoch % opt.save_epoch == 0:
            torch.save(net.state_dict(), '%s/model/netG_%s.pth' % (opt.outf, str(epoch)))

    end = time.time()
    torch.save(net.state_dict(), '%s/model/netG_final.pth' % opt.outf)
    print('Program processed ', end - start, 's, ', (end - start)/60, 'min, ', (end - start)/3600, 'h')
    log.close()