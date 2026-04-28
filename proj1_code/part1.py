#!/usr/bin/python3

from typing import Tuple

import numpy as np

def create_Gaussian_kernel_1D(ksize: int, sigma: int) -> np.ndarray:
    """Create a 1D Gaussian kernel using the specified filter size and standard deviation.
    
    The kernel should have:
    - shape (k,1)
    - mean = floor (ksize / 2)
    - values that sum to 1
    
    Args:
        ksize: length of kernel
        sigma: standard deviation of Gaussian distribution
    
    Returns:
        kernel: 1d column vector of shape (k,1)
    
    HINT:
    - You can evaluate the univariate Gaussian probability density function (pdf) at each
      of the 1d values on the kernel (think of a number line, with a peak at the center).
    - The goal is to discretize a 1d continuous distribution onto a vector.
    """
    k=ksize//2
    d=np.arange(-k,k+1)
    kernel=np.exp(-0.5*(d/sigma)**2)#一维高斯核的公式
    kernel=kernel/kernel.sum()#归一化，使得所有元素的和为1
    kernel=kernel.reshape(-1,1)#将一维数组转换为列向量
    
    return kernel

def create_Gaussian_kernel_2D(cutoff_frequency: int) -> np.ndarray:
    """
    Create a 2D Gaussian kernel using the specified filter size, standard
    deviation and cutoff frequency.

    The kernel should have:
    - shape (k, k) where k = cutoff_frequency * 4 + 1
    - mean = floor(k / 2)
    - standard deviation = cutoff_frequency
    - values that sum to 1

    Args:
        cutoff_frequency: an int controlling how much low frequency to leave in
        the image.
    Returns:
        kernel: numpy nd-array of shape (k, k)

    HINT:
    - You can use create_Gaussian_kernel_1D() to complete this in one line of code.
    - The 2D Gaussian kernel here can be calculated as the outer product of two
      1D vectors. In other words, as the outer product of two vectors, each 
      with values populated from evaluating the 1D Gaussian PDF at each 1d coordinate.
    - Alternatively, you can evaluate the multivariate Gaussian probability 
      density function (pdf) at each of the 2d values on the kernel's grid.
    - The goal is to discretize a 2d continuous distribution onto a matrix.
    """

    sigma = cutoff_frequency#根据cutoff_frequency（截止频率）计算标准差
    ksize = int(4 * sigma + 1)
    if ksize % 2 == 0:
      ksize += 1#确保ksize为奇数，以便有一个中心元素
    kernel_1d = create_Gaussian_kernel_1D(ksize, sigma)
    kernel = np.outer(kernel_1d, kernel_1d)#计算2D高斯核，使用外积将两个1D高斯核(向量）组合成一个2D高斯核（矩阵）

    return kernel


def my_conv2d_numpy(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """Apply a single 2d filter to each channel of an image. Return the filtered image.
    
    Note: we are asking you to implement a very specific type of convolution.
      The implementation in torch.nn.Conv2d is much more general.

    Args:
        image: array of shape (m, n, c)
        filter: array of shape (k, j)
    Returns:
        filtered_image: array of shape (m, n, c), i.e. image shape should be preserved

    HINTS:
    - You may not use any libraries that do the work for you. Using numpy to
      work with matrices is fine and encouraged. Using OpenCV or similar to do
      the filtering for you is not allowed.
    - We encourage you to try implementing this naively first, just be aware
      that it may take an absurdly long time to run. You will need to get a
      function that takes a reasonable amount of time to run so that the TAs
      can verify your code works.
    - If you need to apply padding to the image, only use the zero-padding
      method. You need to compute how much padding is required, if any.
    - "Stride" should be set to 1 in your implementation.
    - You can implement either "cross-correlation" or "convolution", and the result
      will be identical, since we will only test with symmetric filters.
    """

    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    #获取滤波器大小，并计算需要填充的边缘大小
    filter_h, filter_w = filter.shape
    pad_h = filter_h // 2
    pad_w = filter_w // 2
    
    filtered_image = np.zeros_like(image)#创建一个与输入图像大小相同的零矩阵，用于存储滤波后的结果
    
    #判断输入图像是2D(灰度图)还是3D(彩色图)
    if image.ndim == 2:# 灰度图：对高度和宽度进行填充
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')#使用numpy的pad函数对image边缘进行零填充（以便每个像素都能被滤波器覆盖），填充的大小由pad_h（上下）和pad_w（左右）决定，mode='constant'表示使用常数值（0）进行填充
        
        # 滤波器滑动窗口计算
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # 提取与滤波器大小相等的局部区域，进行逐元素相乘后求和.patch是提取的区域
                patch = padded_image[i:i+filter_h, j:j+filter_w]#为什么在padded_image上的起点是 i 和 j？
                #假设pad_h=1（滤波器是 3x3）在 padded_image 中，原图的第一行像素已经被往下挤到了索引 1 的位置。
                #当 i = 0 时，你切片的范围是 0 到 3。这个切片的正中心刚好就是索引 1（也就是原图的 (0,0) 点）。
                filtered_image[i, j] = np.sum(patch * filter)#将提取出来的图像区块和滤波器矩阵进行逐元素相乘,* 不是矩阵乘法，是对应位置相乘,把所有乘积加成一个数字，塞入输出矩阵的对应位置
                
    elif image.ndim == 3:# 彩色图：对高度和宽度填充，通道维度(RGB)不填充
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')#第3维(颜色通道)：0, 0表示完全不填充
        # 需要对 3 个颜色通道分别进行卷积
        for c in range(image.shape[2]):#image.shape[2] 就是通道数（通常是 3）。彩色图片的滤波逻辑是：把红、绿、蓝三个通道拆开，当成三张独立的灰度图来分别处理。 这个 for c 循环就是用来切换当前正在处理的颜色层。
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    patch = padded_image[i:i+filter_h, j:j+filter_w, c]
                    filtered_image[i, j, c] = np.sum(patch * filter)
                    
    return filtered_image
  
  
def my_conv2d_numpy_v2(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """Apply a single 2d filter to each channel of an image. Return the filtered image. Notably, this is the optimized revision of `my_conv2d_numpy()`.
    
    Note: we are asking you to implement a very specific type of convolution.
      The implementation in torch.nn.Conv2d is much more general.

    Args:
        image: array of shape (m, n, c)
        filter: array of shape (k, j)
    Returns:
        filtered_image: array of shape (m, n, c), i.e. image shape should be preserved

    HINTS:
    - You may not use any libraries that do the work for you. Using numpy to
      work with matrices is fine and encouraged. Using OpenCV or similar to do
      the filtering for you is not allowed.
    - We encourage you to try implementing this naively first, just be aware
      that it may take an absurdly long time to run. You will need to get a
      function that takes a reasonable amount of time to run so that the TAs
      can verify your code works.
    - If you need to apply padding to the image, only use the zero-padding
      method. You need to compute how much padding is required, if any.
    - "Stride" should be set to 1 in your implementation.
    - You can implement either "cross-correlation" or "convolution", and the result
      will be identical, since we will only test with symmetric filters.
    """

    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    filter_h, filter_w = filter.shape
    pad_h = filter_h // 2
    pad_w = filter_w // 2
    
    filtered_image = np.zeros_like(image)
    
    if image.ndim == 2:
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')#与之前版本相比，这里的padding方式改成了'reflect'，即反射填充。这种方式会使用图像边缘的像素值来填充，而不是简单地使用0。这可以在某些情况下减少边缘效应，使得滤波结果更自然。
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                patch = padded_image[i:i+filter_h, j:j+filter_w]
                filtered_image[i, j] = np.sum(patch * filter)
                
    elif image.ndim == 3:
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='reflect')#
        for c in range(image.shape[2]):
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    patch = padded_image[i:i+filter_h, j:j+filter_w, c]
                    filtered_image[i, j, c] = np.sum(patch * filter)
                    
    return filtered_image
  


def create_hybrid_image(
    image1: np.ndarray, image2: np.ndarray, filter: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Takes two images and a low-pass filter and creates a hybrid image. Returns
    the low frequency content of image1, the high frequency content of image 2,
    and the hybrid image.

    Args:
        image1: array of dim (m, n, c)
        image2: array of dim (m, n, c)
        filter: array of dim (x, y)
    Returns:
        low_frequencies: array of shape (m, n, c)
        high_frequencies: array of shape (m, n, c)
        hybrid_image: array of shape (m, n, c)

    HINTS:
    - You will use your my_conv2d_numpy() function in this function.
    - You can get just the high frequency content of an image by removing its
      low frequency content. Think about how to do this in mathematical terms.
    - Don't forget to make sure the pixel values of the hybrid image are
      between 0 and 1. This is known as 'clipping'.
    - If you want to use images with different dimensions, you should resize
      them in the notebook code.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]
    assert filter.shape[0] <= image1.shape[0]
    assert filter.shape[1] <= image1.shape[1]
    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1
    #提取图像 1 的低频部分（直接卷积模糊）
    low_frequencies = my_conv2d_numpy(image1, filter)
    
    #提取图像 2 的高频部分（原图减去模糊后的图像）
    smoothed_image2 = my_conv2d_numpy(image2, filter)
    high_frequencies = image2 - smoothed_image2
    
    #混合两张图像（相加）
    hybrid_image = low_frequencies + high_frequencies
    
    #限制像素范围：图像加减可能会产生负数或大于 1 的数，执行截断操作，强行把低于 0 的变成 0，高于 1 的变成 1
    hybrid_image = np.clip(hybrid_image, 0.0, 1.0)
    low_frequencies = np.clip(low_frequencies, 0.0, 1.0)
    high_frequencies = np.clip(high_frequencies, 0.0, 1.0)
    
    return low_frequencies, high_frequencies, hybrid_image