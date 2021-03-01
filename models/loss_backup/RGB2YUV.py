import numpy as np
import torch

def RGB2YUV(input, device):
    b, c, w, h = input.shape
    input2 = (input + 1.) / 2.0
    # print(input)
    matrix = torch.tensor([[0.299, 0.587, 0.114], 
                            [-0.148, 0.289, 0.437],
                            [0.615, 0.515, -0.100]]).to(device)
    input2 = input2.permute(1, 0,  2, 3).reshape(c, b*w*h)
    # print(matrix)
    output = matrix.mm(input2)
    # print(output)
    output = output.reshape(c, b, w, h).permute(1, 0, 2, 3)
    # print(output)
    output = (output - 0.5) / 0.5
    return output

# a = torch.ones(2, 3)
# b = torch.zeros(2, 3, 5, 6)
# print(torch.matmul(a, b))

# a = torch.ones(3, 3)
# b = torch.arange(24).reshape(2, 3, 2, 2).float() # b c w h
# print(type(a))
# print(type(b))
# c = b.permute(1, 0, 2, 3)
# d = c.reshape(3, 2*2*2)
# print(b)
# print(c)
# print(d)
# print(a.shape)
# print(d.shape)
# e = a.mm(d)
# print(e)
# f = e.reshape(3, 2, 2, 2).permute(1, 0, 2, 3)
# print(f)

