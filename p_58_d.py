import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy.ndimage import generic_filter

# 设置字体以防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取图像文件
image = io.imread('d:/桌面/数字图像处理实验/chapter5/Fig0508salt.tif')

# 定义逆谐波均值滤波器函数
def contraharmonic_mean_filter(pixels, Q):
    numerator = np.sum(pixels ** (Q + 1))
    denominator = np.sum(pixels ** Q)
    return numerator / (denominator + 1e-9)  # 加上一个小值以避免除零问题

# 使用 3x3、阶数为 1.5 的逆谐波均值滤波器对图像进行滤波
Q = 1.5
filtered_image_contraharmonic = generic_filter(image, contraharmonic_mean_filter, size=3, extra_arguments=(Q,))

# 显示原始图像和滤波后的图像
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("原始图像")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("逆谐波均值滤波后的图像")
plt.imshow(filtered_image_contraharmonic, cmap='gray')
plt.axis('off')

plt.show()