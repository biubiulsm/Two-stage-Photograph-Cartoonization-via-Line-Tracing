import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

def weight_calculate(device, I, H, dr):
    b, c, h, w = I.shape
    new_I = F.pad(I, pad=[H//2, H//2, H//2, H//2])
    Map = torch.tensor([]).to(device)
    for i in range(H):
        map = torch.tensor([]).to(device)
        for j in range(H):
            map = torch.cat((map, torch.exp(-((I - new_I[:, :, i:(i+h), j:(j+w)])**2).sum(dim=1) /(2* dr**2)).unsqueeze(1)), 1)
        Map = torch.cat((Map, map.unsqueeze(1)), 1)
    return Map

def flattening_loss_Gause(device, T, Map, H, p, Kernel):
    b, c, h, w = T.shape
    new_T = F.pad(T, pad=[H//2, H//2, H//2, H//2])
    loss = torch.tensor(0.0).to(device)
    for i in range(H):
        for j in range(H):
            loss += torch.sum(torch.pow(torch.abs(T - new_T[:, :, i:(i+h), j:(j+w)]), p).sum(dim=1).mul(Map[:, i, j, :, :]) * Kernel[0, i, j])
    loss = loss / (h*w*b*1.)
    return loss


def Flattening_Loss2_Gause(device, I, T, H, dr, p, Kernel):
    Map = weight_calculate(device, I, H, dr)
    loss = flattening_loss_Gause(device, T, Map, H, p, Kernel)
    return loss


'''for testing flattening loss'''
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# b = torch.randn(1, 150).float().to(device)
# b = b.reshape([2, 3, 5, 5])
# H = 3
# dr = 0.1
# p = 2
# T = torch.arange(0, 150).float().to(device).reshape([2, 3, 5, 5]) * 0.1
# Map = weight_calculate(device, b, H, dr)
# loss = flattening_loss(device, T, Map, H, p)
# print(Map.shape)
# print(loss)