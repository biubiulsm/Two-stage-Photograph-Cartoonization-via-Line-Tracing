import torch

def getAngle(a, b, batch_size):
    a = a.view(batch_size, 128, -1)
    b = b.view(batch_size, 128, -1)
    inner_product = (a * b).sum(dim=1)
    a_norm = a.pow(2).sum(dim=1).pow(0.5)
    b_norm = b.pow(2).sum(dim=1).pow(0.5)
    cos = inner_product / (2 * a_norm * b_norm)
    angle = torch.acos(cos)

    return angle.sum(dim=1, keepdim=True)
    

# a = torch.ones(2, 6, 256, 256)
# b = torch.ones(2, 6, 256, 256)

# print(getAngle(a, b, 2))