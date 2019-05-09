from __future__ import print_function
import numpy as np

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)
 
def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
 
def compute_per_iou(pred, label, cls_nums, hist):
    hist += fast_hist(label.flatten(), pred.flatten(), cls_nums)
    mIoUs = per_class_iu(hist)
    mIoU = np.nanmean(mIoUs)
    return mIoUs, mIoU

if __name__ == "__main__":
    hist = np.zeros((2,2))
    a = np.array([[[10,10],[10,10]],[[1,1],[0,0]],[[0,0],[0,0]]],dtype=np.int32)
    b = np.array([[[1,10],[11,0]],[[10,1],[0,0]],[[0,0],[1,1]]],dtype=np.int32)
    a = np.minimum(a,1)
    b = np.minimum(b,1)
    print(a.flatten(),b.flatten())
    print(compute_per_iou(a,b,2,hist)[0][1])