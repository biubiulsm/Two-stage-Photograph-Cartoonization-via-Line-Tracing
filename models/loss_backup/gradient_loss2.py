from PIL import Image
import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms
from torch.autograd import Variable



def get_optics(x, device):
    # x = img[:, 0, :, :] * 0.299 + img[:, 1, :, :] * 0.587 +img[:, 2, :, :] * 0.114
    # x = torch.unsqueeze(x, 1)
    # print(x.shape)
    a=np.array([[-1, 0, 1],[-2,0,2],[-1,0,1]])
    # a = np.array([a, a, a])
    conv1=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    a = torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0).to(device)
    conv1.weight=nn.Parameter(a)
    conv1.weight.requires_grad = False
    G_x=conv1(x)
    

    b=np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])
    # b = np.array([b, b, b])
    conv2=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv2.weight=nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0).to(device))
    conv2.weight.requires_grad = False
    G_y=conv2(x)
    
    # # can't renew
    # G_x.requires_grad = False
    # G_y.requires_grad = False
    return G_x, G_y
