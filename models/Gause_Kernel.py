import numpy as np
import torch
import math
# import cv2 # python3.6
# def Gause_Kernel(N, batchsize, es):
#     a = cv2.getGaussianKernel(N, es)
#     return a

# a = Gause_Kernel(3, 1, 1.)
# print(a)
# print(type(a)) # numpy.ndarray

def Gause_Kernel(N, batchsize, es):
    dis = N//2
    kernel = np.zeros((N, N))
    for i in range(-dis, dis):
        for j in range(-dis, dis):
            kernel[i, j] = math.exp(-(i**2 + j**2)/(2*es*es))
    print(kernel.shape)
    kernel = np.tile(kernel, (batchsize, 1, 1))
    Kernel = torch.from_numpy(kernel)
    return Kernel

# a = Gause_Kernel(3, 2, 0.2)
# print(a.shape)
