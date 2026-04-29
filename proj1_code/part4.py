#!/usr/bin/python3

from typing import Tuple

import numpy as np

import numpy.fft as fft

from utils import load_image, save_image, PIL_resize, numpy_arr_to_PIL_image, PIL_image_to_numpy_arr, im2single, single2im

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    计算两张图像的 PSNR (Peak Signal-to-Noise Ratio)
    公式: PSNR = 10 * log10(MAX^2 / MSE)
    """
    # 计算均方误差 MSE
    mse = np.mean(np.square(img1 - img2))
    
    # 如果两张图完全一样，MSE为0，PSNR趋近于无穷大
    if mse == 0:
        return float('inf')
    
    # 因为通过 load_image 读取的图像像素值已被归一化到 [0.0, 1.0] 区间
    # 所以这里的 MAX 就是 1.0
    max_pixel = 1.0 
    
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

def frequency_compression(image: np.ndarray, retention_ratio: float) -> np.ndarray:
    """
    Task 1: 使用 2D FFT 进行频域压缩
    """
    # 兼容单通道(灰度)和多通道(彩色)图像的处理
    is_grayscale = False
    if image.ndim == 2:
        is_grayscale = True
        # 给灰度图临时增加一个通道维度，变成 (H, W, 1)，方便后面用统一的循环处理
        image = np.expand_dims(image, axis=2)
        
    H, W, C = image.shape
    compressed_image = np.zeros_like(image, dtype=float)
    
    # 计算需要保留的中心低频区域大小 (矩形 Mask)
    keep_h = int(H * retention_ratio)
    keep_w = int(W * retention_ratio)
    
    # 计算 Mask 的边界索引
    start_h = (H - keep_h) // 2
    end_h = start_h + keep_h
    start_w = (W - keep_w) // 2
    end_w = start_w + keep_w
    
    # 对每个颜色通道独立进行傅里叶变换
    for c in range(C):
        channel = image[:, :, c]
        
        # 1. 2D Fourier Transform (傅里叶正变换)
        f_transform = fft.fft2(channel)
        
        # 2. Shift (将低频分量移动到矩阵中心)
        f_shift = fft.fftshift(f_transform)
        
        # 3. Create and apply low-pass filter (创建矩形低通掩膜并过滤)
        mask = np.zeros((H, W))
        mask[start_h:end_h, start_w:end_w] = 1.0
        f_shift_filtered = f_shift * mask
        
        # 4. Inverse shift (将中心低频移回矩阵左上角的标准位置)
        f_ishift = fft.ifftshift(f_shift_filtered)
        
        # 5. Inverse 2D Fourier Transform (逆傅里叶变换，重建回空间域图像)
        reconstructed_channel = fft.ifft2(f_ishift)
        
        # FFT 逆变换的结果可能带有极微小的复数虚部，我们只取实数部分 (np.real)
        compressed_image[:, :, c] = np.real(reconstructed_channel)
        
    # 裁剪溢出值，防止傅里叶重建引起的数值越界 (比如出现 -0.01 或 1.02)
    compressed_image = np.clip(compressed_image, 0.0, 1.0)
    
    # 如果原图是灰度图，要把刚才增加的维度再去掉，恢复成 (H, W)
    if is_grayscale:
        compressed_image = np.squeeze(compressed_image, axis=2)
        
    return compressed_image

if __name__ == "__main__":
    # 这里的代码只是为了方便你在本地运行获取报告数据，不会影响评测
    import os
    
    # 设定测试图片的相对路径
    image_path = '../data/1a_dog.bmp' 
    
    if os.path.exists(image_path):
        original_image = load_image(image_path)
        retention_ratios = [0.1, 0.3, 0.5, 0.7]
        
        print(f"==== 频域压缩测试 ====\n目标图像: {image_path}")
        for ratio in retention_ratios:
            # 压缩
            compressed_img = frequency_compression(original_image, ratio)
            # 计算并输出 PSNR
            psnr_value = calculate_psnr(original_image, compressed_img)
            print(f"保留率 (Retention Ratio): {ratio} | PSNR: {psnr_value:.2f} dB")
            
            # 取消下面这行的注释，可以把压缩后的图片保存到 results 文件夹里放进 PPT
            # save_image(f'../results/compressed_ratio_{ratio}.jpg', compressed_img)
    else:
        print(f"未找到图片：{image_path}，请确保你在 proj1_code 文件夹下运行此脚本。")