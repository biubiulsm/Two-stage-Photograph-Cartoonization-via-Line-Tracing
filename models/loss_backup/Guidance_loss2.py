import torch.nn as nn
import torch
import numpy as np

import torch.nn.functional as F

def get_guidance(device, x, H):
    b, c, h, w = x.shape
    new_x = F.pad(x, pad=[H//2, H//2, H//2, H//2])
    Map = torch.tensor([]).to(device)
    for i in range(H):
        map = torch.tensor([]).to(device)
        for j in range(H):
            map = torch.cat((map, torch.abs(torch.sum(x - new_x[:, :, i:(i+h), j:(j+w)], 1)).unsqueeze(1)), 1)
        Map = torch.cat((Map, map.unsqueeze(1)), 1)
    Map2 = torch.sum(torch.sum(Map, 1), 1)
    return Map2