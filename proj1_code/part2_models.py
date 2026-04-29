#!/usr/bin/python3

"""
PyTorch tutorial on constructing neural networks:
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from proj1_code.part1 import create_Gaussian_kernel_2D


class HybridImageModel(nn.Module):
    def __init__(self):
        """
        Initializes an instance of the HybridImageModel class.
        """
        super(HybridImageModel, self).__init__()

    def get_kernel(self, cutoff_frequency: int) -> torch.Tensor:
        """
        Returns a Gaussian kernel using the specified cutoff frequency.

        PyTorch requires the kernel to be of a particular shape in order to
        apply it to an image. Specifically, the kernel needs to be of shape
        (c, 1, k, k) where c is the # channels in the image. Start by getting a
        2D Gaussian kernel using your implementation from Part 1, which will be
        of shape (k, k). Then, let's say you have an RGB image, you will need to
        turn this into a Tensor of shape (3, 1, k, k) by stacking the Gaussian
        kernel 3 times.

        Args
            cutoff_frequency: int specifying cutoff_frequency
        Returns
            kernel: Tensor of shape (c, 1, k, k) where c is # channels

        HINTS:
        - You will use the create_Gaussian_kernel_2D() function from part1.py in
          this function.
        - Since the # channels may differ across each image in the dataset,
          make sure you don't hardcode the dimensions you reshape the kernel
          to. There is a variable defined in this class to give you channel
          information.
        - You can use np.reshape() to change the dimensions of a numpy array.
        - You can use np.tile() to repeat a numpy array along specified axes.
        - You can use torch.Tensor() to convert numpy arrays to torch Tensors.
        """

        ############################
        ### TODO: YOUR CODE HERE ###

        # 调用 Part1 的函数生成 2D 高斯核
        kernel_2d = create_Gaussian_kernel_2D(cutoff_frequency)
        k = kernel_2d.shape[0]# 获取核的尺寸

        # 使用 np.reshape 改变维度为 (1, 1, k, k)因为PyTorch 的 F.conv2d 要求卷积核的形状为 (输出通道, 每个组的输入通道, 高, 宽)
        kernel_reshaped = np.reshape(kernel_2d, (1, 1, k, k))

        # 使用 np.tile 沿着通道维度(第0维)复制 self.n_channels 次，变为 (c, 1, k, k)：需要 3 个独立的滤波器（输出通道=3），每个滤波器只负责看 1 种颜色（输入通道/groups=1），它们的大小是 k×k
        kernel_tiled = np.tile(kernel_reshaped, (self.n_channels, 1, 1, 1))#参数1意味着该参量保持不变，n意味着复制n份

        # 使用 torch.Tensor() 转换为 PyTorch 的 Tensor
        kernel = torch.Tensor(kernel_tiled)

        ### END OF STUDENT CODE ####
        ############################

        return kernel

    def low_pass(self, x: torch.Tensor, kernel: torch.Tensor):
        """
        Applies low pass filter to the input image.

        Args:
            x: Tensor of shape (b, c, m, n) where b is batch size
            kernel: low pass filter to be applied to the image
        Returns:
            filtered_image: Tensor of shape (b, c, m, n)

        HINTS:
        - You should use the 2d convolution operator from torch.nn.functional.
        - Make sure to pad the image appropriately (it's a parameter to the
          convolution function you should use here!).
        - Pass self.n_channels as the value to the "groups" parameter of the
          convolution function. This represents the # of channels that the
          filter will be applied to.
        """

        ############################
        ### TODO: YOUR CODE HERE ###

        # 计算需要的 padding 大小 (滤波器尺寸的一半)
        # 此时 kernel 的 shape 是 (c, 1, k, k)，所以取最后一个维度即可
        pad_size = kernel.shape[-1] // 2

        # 调用 F.conv2d。注意：必须传入 groups=self.n_channels
        # 这是为了确保 RGB 三个通道分别与自己的滤波层卷积，而不发生颜色交叉混合
        filtered_image = F.conv2d(x, kernel, padding=pad_size, groups=self.n_channels)#加上 groups=3会把输入的 3 层通道（颜色通道）拆成 3 个独立的单层图

        ### END OF STUDENT CODE ####
        ############################

        return filtered_image

    def forward(
        self, image1: torch.Tensor, image2: torch.Tensor, cutoff_frequency: torch.Tensor
    ):
        """
        Takes two images and creates a hybrid image. Returns the low frequency
        content of image1, the high frequency content of image 2, and the
        hybrid image.

        Args:
            image1: Tensor of shape (b, c, m, n)
            image2: Tensor of shape (b, c, m, n)
            cutoff_frequency: Tensor of shape (b)
        Returns:
            low_frequencies: Tensor of shape (b, c, m, n)
            high_frequencies: Tensor of shape (b, c, m, n)
            hybrid_image: Tensor of shape (b, c, m, n)

        HINTS:
        - You will use the get_kernel() function and your low_pass() function
          in this function.
        - Similar to Part 1, you can get just the high frequency content of an
          image by removing its low frequency content.
        - Don't forget to make sure to clip the pixel values >=0 and <=1. You
          can use torch.clamp().
        - If you want to use images with different dimensions, you should
          resize them in the HybridImageDataset class using
          torchvision.transforms.
        """
        self.n_channels = image1.shape[1]

        ############################
        ### TODO: YOUR CODE HERE ###

        # DataLoader（数据加载器）打包后，原本的一个普通数字（比如 7）会被包装成一个张量（Tensor）
        # .item()提取出标量值用于生成 kernel，把张量里包裹的唯一数字“剥”出来，变成普通的 Python 数字
        cf = int(cutoff_frequency[0].item()) if cutoff_frequency.dim() > 0 else int(cutoff_frequency.item())

        # 获取处理好的四维卷积核，并.to确保它和输入的图像在同一个设备上 (CPU 或 GPU)否则会报错（PyTorch中张量必须在同一设备上运算）
        kernel = self.get_kernel(cf).to(image1.device)

        # 求 image1 的低频图像
        low_frequencies = self.low_pass(image1, kernel)

        # 求 image2 的高频图像 (原图2 减去 图2的低频部分)
        high_frequencies = image2 - self.low_pass(image2, kernel)

        # 将高频和低频混合，并使用 torch.clamp 进行 [0.0, 1.0] 的截断操作
        hybrid_image = torch.clamp(low_frequencies + high_frequencies, 0.0, 1.0)

        ### END OF STUDENT CODE ####
        ############################

        return low_frequencies, high_frequencies, hybrid_image
