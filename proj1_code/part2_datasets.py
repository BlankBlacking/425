#!/usr/bin/python3

"""
PyTorch tutorial on data loading & processing:
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

import os
from typing import List, Tuple

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms


def make_dataset(path: str) -> Tuple[List[str], List[str]]:
    """
    Creates a dataset of paired images from a directory.

    The dataset should be partitioned into two sets: one contains images that
    will have the low pass filter applied, and the other contains images that
    will have the high pass filter applied.

    Args:
        path: string specifying the directory containing images
    Returns:
        images_a: list of strings specifying the paths to the images in set A,
           in lexicographically-sorted order
        images_b: list of strings specifying the paths to the images in set B,
           in lexicographically-sorted order
    """
    ############################
    ### TODO: YOUR CODE HERE ###

    # 获取目录下所有文件并按字母顺序排序
    files = sorted([f for f in os.listdir(path) if not f.startswith('.')])
    
    # 根据/data的命名规律筛选需要的图片
    images_a = [os.path.join(path, f) for f in files if 'a_' in f]
    images_b = [os.path.join(path, f) for f in files if 'b_' in f]

    ### END OF STUDENT CODE ####
    ############################
    return images_a, images_b


def get_cutoff_frequencies(path: str) -> List[int]:
    """
    Gets the cutoff frequencies corresponding to each pair of images.

    The cutoff frequencies are the values you discovered from experimenting in
    part 1.

    Args:
        path: string specifying the path to the .txt file with cutoff frequency
        values
    Returns:
        cutoff_frequencies: numpy array of ints. The array should have the same
            length as the number of image pairs in the dataset
    """
    ############################
    ### TODO: YOUR CODE HERE ###
    # 读取 cutoff_frequencies.txt，将每一行转换为整数
    with open(path, 'r') as f:
        cf_list = [int(line.strip()) for line in f.readlines() if line.strip()]
    cutoff_frequencies = np.array(cf_list)
    ### END OF STUDENT CODE ####
    ############################
    return cutoff_frequencies


class HybridImageDataset(data.Dataset):
    """Hybrid images dataset."""

    def __init__(self, image_dir: str, cf_file: str) -> None:
        """
        HybridImageDataset class constructor.

        You must replace self.transform with the appropriate transform from
        torchvision.transforms that converts a PIL image to a torch Tensor. You
        can specify additional transforms (e.g. image resizing) if you want to,
        but it's not necessary for the images we provide you since each pair has
        the same dimensions.

        Args:
            image_dir: string specifying the directory containing images
            cf_file: string specifying the path to the .txt file with cutoff
            frequency values
        """
        images_a, images_b = make_dataset(image_dir)
        cutoff_frequencies = get_cutoff_frequencies(cf_file)

        self.transform = None
        ############################
        ### TODO: YOUR CODE HERE ###

        # ToTensor()将从传统的图片库（如OpenCV, PIL, NumPy）读取的图片格式 (高度H, 宽度W, 颜色通道C)，像素值0~255转换为PyTorch 模型强制要求格式：(颜色通道C, 高度H, 宽度W)，像素值归一化
        self.transform = transforms.ToTensor()

        ### END OF STUDENT CODE ####
        ############################

        self.images_a = images_a
        self.images_b = images_b
        self.cutoff_frequencies = cutoff_frequencies

    def __len__(self) -> int:
        """Returns number of pairs of images in dataset."""

        ############################
        ### TODO: YOUR CODE HERE ###

        return len(self.images_a)
    
        ### END OF STUDENT CODE ####
        ############################

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns the pair of images and corresponding cutoff frequency value at
        index `idx`.

        Since self.images_a and self.images_b contain paths to the images, you
        should read the images here and normalize the pixels to be between 0
        and 1. Make sure you transpose the dimensions so that image_a and
        image_b are of shape (c, m, n) instead of the typical (m, n, c), and
        convert them to torch Tensors.

        Args:
            idx: int specifying the index at which data should be retrieved
        Returns:
            image_a: Tensor of shape (c, m, n)
            image_b: Tensor of shape (c, m, n)
            cutoff_frequency: int specifying the cutoff frequency corresponding
               to (image_a, image_b) pair

        HINTS:
        - You should use the PIL library to read images
        - You will use self.transform to convert the PIL image to a torch Tensor
        """

        ############################
        ### TODO: YOUR CODE HERE ###

        # 使用 PIL 读取图像并确保是 RGB 模式（强制把可能存在的灰度图或带透明度的四通道图全部变成标准的 3 通道彩色图）
        img_a = PIL.Image.open(self.images_a[idx]).convert('RGB')
        img_b = PIL.Image.open(self.images_b[idx]).convert('RGB')
        
        # 使用我们在 __init__ 中定义的 transform 对图片格式进行转换
        image_a = self.transform(img_a)
        image_b = self.transform(img_b)
        
        # 获取对应的频率
        cutoff_frequency = self.cutoff_frequencies[idx]

        ### END OF STUDENT CODE ####
        ############################

        return image_a, image_b, cutoff_frequency