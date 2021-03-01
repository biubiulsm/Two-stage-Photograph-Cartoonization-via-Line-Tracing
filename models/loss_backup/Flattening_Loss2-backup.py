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

def flattening_lossSmooth(device, T, Map, H, p, BinaryMap):
    b, c, h, w = T.shape
    new_T = F.pad(T, pad=[H//2, H//2, H//2, H//2])
    loss = torch.tensor(0.0).to(device)
    for i in range(H):
        for j in range(H):
            # print(((T - new_T[:, :, i:(i+h), j:(j+w)]) ** p).sum(dim=1))
            # print(Map[:, i, j, :, :])
            # print(torch.pow(torch.abs(T - new_T[:, :, i:(i+h), j:(j+w)]), p).shape)
            loss += torch.sum(torch.pow(torch.abs(T - new_T[:, :, i:(i+h), j:(j+w)]), p).sum(dim=1).mul(Map[:, i, j, :, :]).mul(BinaryMap))
    loss = loss / (h*w*b*1.)
    return loss

def Flattening_Loss2Smooth(device, I, T, H, dr, p, BinaryMap):
    Map = weight_calculate(device, I, H, dr)
    loss = flattening_lossSmooth(device, T, Map, H, p, BinaryMap)
    return loss


def flattening_loss(device, T, Map, H, p):
    b, c, h, w = T.shape
    new_T = F.pad(T, pad=[H//2, H//2, H//2, H//2])
    loss = torch.tensor(0.0).to(device)
    for i in range(H):
        for j in range(H):
            # print(((T - new_T[:, :, i:(i+h), j:(j+w)]) ** p).sum(dim=1))
            # print(Map[:, i, j, :, :])
            # print(torch.pow(torch.abs(T - new_T[:, :, i:(i+h), j:(j+w)]), p).shape)
            loss += torch.sum(torch.pow(torch.abs(T - new_T[:, :, i:(i+h), j:(j+w)]), p).sum(dim=1).mul(Map[:, i, j, :, :]))
    loss = loss / (h*w*b*1.)
    return loss


def Flattening_Loss2(device, I, T, H, dr, p):
    Map = weight_calculate(device, I, H, dr)
    loss = flattening_loss(device, T, Map, H, p)
    return loss


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

def flattening_loss_Gause_reverse(device, T, Map, I, H, p, Kernel):
    b, c, h, w = T.shape
    loss = torch.tensor(0.0).to(device)
    for i in range(H):
        for j in range(H):
            loss += torch.sum(torch.pow(torch.abs(T - I), p).sum(dim=1).mul(Map[:, i, j, :, :]) * Kernel[0, i, j])
    loss = loss / (h*w*b*1.)
    return loss

def Flattening_Loss2_Gause_reverse(device, I, I_ori, T, H, dr, p, Kernel):
    Map = weight_calculate(device, I, H, dr)
    Map = torch.pow(Map, dr**5)
    loss = flattening_loss_Gause_reverse(device, T, Map, I_ori, H, p, Kernel)
    return loss



def Flattening_Loss2_Gause_limit(device, I, T, T2, H, dr, dr2, p, Kernel, Kernel2):
    Map = weight_calculate(device, I, H, dr)
    Map2 = weight_calculate(device, I, H, dr2)
    b, c, h, w = T.shape
    new_T = F.pad(T, pad=[H//2, H//2, H//2, H//2])
    new_T2 = F.pad(T2, pad=[H//2, H//2, H//2, H//2])
    loss = torch.tensor(0.0).to(device)
    for i in range(H):
        for j in range(H):
            t1 = torch.pow(torch.abs(T - new_T[:, :, i:(i+h), j:(j+w)]), p).sum(dim=1)
            t2 = torch.pow(torch.abs(T2 - new_T2[:, :, i:(i+h), j:(j+w)]), p).sum(dim=1)
            # print(torch.min(torch.abs(Map[:, i, j, :, :] * Kernel[0, i, j] - Map2[:, i, j, :, :] * Kernel2[0, i, j])))
            loss += torch.sum(torch.exp(-(t2-t1)*(Map[:, i, j, :, :] * Kernel[0, i, j] - Map2[:, i, j, :, :] * Kernel2[0, i, j])).mul(torch.abs(Map[:, i, j, :, :] * Kernel[0, i, j] - Map2[:, i, j, :, :] * Kernel2[0, i, j])))
    loss = loss / (h*w*b*1.)
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