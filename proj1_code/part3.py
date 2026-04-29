#!/usr/bin/python3

"""
PyTorch tutorial on constructing neural networks:
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def my_conv2d_pytorch(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Applies input filter(s) to the input image.

    Args:
        image: Tensor of shape (1, d1, h1, w1)
        kernel: Tensor of shape (N, d1/groups, k, k) to be applied to the image
    Returns:
        filtered_image: Tensor of shape (1, d2, h2, w2) where
           d2 = N
           h2 = (h1 - k + 2 * padding) / stride + 1
           w2 = (w1 - k + 2 * padding) / stride + 1

    HINTS:
    - You should use the 2d convolution operator from torch.nn.functional.
    - In PyTorch, d1 is `in_channels`, and d2 is `out_channels`
    - Make sure to pad the image appropriately (it's a parameter to the
      convolution function you should use here!).
    - You can assume the number of groups is equal to the number of input channels.
    - You can assume only square filters for this function.
    """

    ############################
    ### TODO: YOUR CODE HERE ###
    #image: Tensor of shape (1本次运算所处理图片数量, d1输入通道数, h1, w1)
    #kernel: Tensor of shape (N输出通道数（输出几层图）, d1/groups每个滤波器负责的输入通道数, k, k)
    # 因为groups=d1,使用了深度可分离卷积，所以每个滤波器只负责看 1 个输入通道（颜色通道），输出一个单层图。最终输出的图层数等于滤波器的数量 N。
    in_channels = image.shape[1]
    
    pad_size = kernel.shape[2] // 2#计算填充尺寸
    
    # 调用强大的 PyTorch 底层卷积算子 F.conv2d
    filtered_image = F.conv2d(image, kernel, padding=pad_size, groups=in_channels)

    ### END OF STUDENT CODE ####
    ############################

    return filtered_image
