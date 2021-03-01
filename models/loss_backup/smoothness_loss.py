import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable


# def total_variation(x):
#     b, ch, h, w = x.data.shape
#     wh = Variable(xp.asarray([[[[1], [-1]], [[0], [0]], [[0], [0]]], [[[0], [0]], [[1], [-1]], [[0], [0]]], [[[0], [0]], [[0], [0]], [[1], [-1]]]], dtype=np.float32), volatile=x.volatile)
#     ww = Variable(xp.asarray([[[[1, -1]], [[0, 0]], [[0, 0]]], [[[0, 0]], [[1, -1]], [[0, 0]]], [[[0, 0]], [[0, 0]], [[1, -1]]]], dtype=np.float32), volatile=x.volatile)
#     return F.sum(F.convolution_2d(x, W=wh) ** 2) + F.sum(F.convolution_2d(x, W=ww) ** 2)

def smoothness(x):
    a = np.array([[[[1], [-1]], [[0], [0]], [[0], [0]]], [[[0], [0]], [[1], [-1]], [[0], [0]]], [[[0], [0]], [[0], [0]], [[1], [-1]]]])
    conv1 = nn.Conv2d(3, 3, kernel_size=[3, 3, 2, 1], bias=False)
    a = torch.from_numpy(a).float().cuda()
    conv1.weight = nn.Parameter(a)
    conv1.weight.requires_grad = False

    b = np.array([[[[1, -1]], [[0, 0]], [[0, 0]]], [[[0, 0]], [[1, -1]], [[0, 0]]], [[[0, 0]], [[0, 0]], [[1, -1]]]])
    conv2 = nn.Conv2d(3, 3, kernel_size=[3, 3, 1, 2], bias=False)
    b = torch.from_numpy(b).float().cuda()
    conv2.weight = nn.Parameter(b)
    conv2.weight.requires_grad = False
    
    return torch.sum(conv1(x)**2) + torch.sum(conv2(x)**2)