# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 18:32:14 2020

@author: Woody
"""

# 导入需要的包
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 读入图片
image = Image.open('lotus.jpg')
image = np.array(image)
# 查看数据形状，其形状是[H, W, 3]，
# 其中H代表高度， W是宽度，3代表RGB三个通道 
print(image.shape) # (480, 320, 3)

# 原始图片
plt.imshow(image)

# 垂直方向翻转
# 这里使用数组切片的方式来完成，
# 相当于将图片最后一行挪到第一行，
# 倒数第二行挪到第二行，..., 
# 第一行挪到倒数第一行
# 对于行指标，使用::-1来表示切片，
# 负数步长表示以最后一个元素为起点，向左走寻找下一个点
# 对于列指标和RGB通道，仅使用:表示该维度不改变
image1 = image[::-1, :, :]
plt.imshow(image1)

# 水平方向翻转
image2 = image[:, ::-1, :]
plt.imshow(image2)

# 保存图片
im2 = Image.fromarray(image2)
im2.save('im2.jpg')

#  高度方向裁剪
H, W = image.shape[0], image.shape[1]
# 注意此处用整除，H_start必须为整数
H1 = H // 2 
H2 = H
image3 = image[H1:H2, :, :]
plt.imshow(image3)

#  宽度方向裁剪
W1 = W//6
W2 = W//3 * 2
image4 = image[:, W1:W2, :]
plt.imshow(image4)

# 两个方向同时裁剪
image5 = image[H1:H2, \
               W1:W2, :]
plt.imshow(image5)

# 调整亮度
image6 = image * 0.5
plt.imshow(image6.astype('uint8'))

# 调整亮度
image7 = image * 2.0
# 由于图片的RGB像素值必须在0-255之间，
# 此处使用np.clip进行数值裁剪
image7 = np.clip(image7, \
        a_min=None, a_max=255.)
plt.imshow(image7.astype('uint8'))

#高度方向每隔一行取像素点
image8 = image[::2, :, :]
plt.imshow(image8)


#宽度方向每隔一列取像素点
image9 = image[:, ::2, :]
plt.imshow(image9)

#间隔行列采样，图像尺寸会减半，清晰度变差
image10 = image[::2, ::2, :]
plt.imshow(image10)
print(image10.shape) # (240, 160, 3)
