import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from skimage import io
from scipy.ndimage import generic_filter

# 设置字体以防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取图像文件
image = io.imread('d:/桌面/数字图像处理实验/chapter5/Fig0507gauss.tif')

# 使用 3x3 的算术均值滤波器对图像进行滤波
filtered_image_arithmetic = uniform_filter(image, size=3)

# 定义几何均值滤波器函数
def geometric_mean_filter(pixels):
    return np.exp(np.mean(np.log(pixels + 1e-9)))  # 加上一个小值以避免对数零的问题

# 使用 3x3 的几何均值滤波器对图像进行滤波
filtered_image_geometric = generic_filter(image, geometric_mean_filter, size=3)

# 显示原始图像、算术均值滤波后的图像和几何均值滤波后的图像
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("原始图像")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("算术均值滤波后的图像")
plt.imshow(filtered_image_arithmetic, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("几何均值滤波后的图像")
plt.imshow(filtered_image_geometric, cmap='gray')
plt.axis('off')

plt.show()