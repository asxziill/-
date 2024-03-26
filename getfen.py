# 导入所需模块
import cv2
from matplotlib import pyplot as plt
import os
import numpy as np


import testFUN as fun



# path = './c1.jpg'


path = './c5.jpg'

# 读取待检测图片
origin_image = cv2.imread(path)
# plt_show0(origin_image)
# 复制一张图片，在复制图上进行图像操作，保留原图
image = origin_image.copy()



image = fun.getSide(image)
# fun.plt_show(image)

image = fun.getRange(image)
# fun.plt_show(image)



image_list = fun.splitRange(image, origin_image)

import tate
fun.plt_show(image_list[0])
tate.recoginse_text(image_list[0])
# image = fun.getSimpleRange(image_list[0])
# fun.plt_show0(image)

# chinese_words_list = fun.get_chinese_words_list()
# print(chinese_words_list)

# for img in image_list:
#     fun.plt_show0(img)


# # 进一步分割
# #车牌字符分割
# # 图像去噪灰度处理
# gray_image = gray_guss(image)
# # 图像阈值化操作——获得二值化图   
# ret, image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)


# #膨胀操作，使“苏”字膨胀为一个近似的整体，为分割做准备
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
# image = cv2.dilate(image, kernel)

# # 膨胀后的图像
# # plt_show(image)

# # 查找轮廓
# contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # debugcontours(contours, image)
# # words 词图矩形
# words = []
# word_images = []
# #对所有轮廓逐一操作
# for item in contours:
#     word = []
#     rect = cv2.boundingRect(item)
#     x = rect[0]
#     y = rect[1]
#     weight = rect[2]
#     height = rect[3]
#     word.append(x)
#     word.append(y)
#     word.append(weight)
#     word.append(height)
#     words.append(word)
# # 排序，车牌号有顺序。words是一个嵌套列表
# # 按x排序
# words = sorted(words,key=lambda s:s[0],reverse=False)

# i = 0
# #word中存放轮廓的起始点和宽高
# # 筛选
# for word in words:
#     # 筛选字符的轮廓
#     if (word[3] > (word[2] * 1.5)) and (word[3] < (word[2] * 3.5)) and (word[2] > 25):
#         i = i+1
#         splite_image = image[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
#         word_images.append(splite_image)
#         # print(i)
# # print(words)

# # 展示图片 下标和编号
# # for i,j in enumerate(word_images):  
# #     plt.subplot(1,7,i+1)
# #     plt.imshow(word_images[i],cmap='gray')
# # plt.show()

