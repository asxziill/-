import cv2
import pytesseract
import matplotlib.pyplot as plt
import numpy as np

def show(image, str = 'Image'):
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

imagepath = "t1.jpg"



    
image =  cv2.imread(imagepath)

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# edges = cv2.Canny(image, threshold1, threshold2)
# threshold1和threshold2是两个阈值参数。
# 较大的阈值用于检测强边缘，较小的阈值用于检测弱边缘。
# 像素梯度高于threshold2的被认为是强边缘，低于threshold1的被认为是非边缘，介于两者之间的被认为是弱边缘。
# Canny函数返回一个二值图像，其中边缘像素为白色，非边缘像素为黑色。

# 边缘检测
edges = cv2.Canny(gray, 50, 150)
# show(edges, 'ee')

# lines = cv2.HoughLinesP(image, rho, theta, threshold, minLineLength, maxLineGap)
# image：输入的二值图像，通常是通过其他方法（如边缘检测）得到的。
# rho：以像素为单位的距离精度，一般设置为1。
# theta：以弧度为单位的角度精度，一般设置为np.pi/180。
# threshold：直线被接受的最小投票数，即直线被认为是有效直线的最小像素数。
# minLineLength：被认为是线段的最小长度。
# maxLineGap：同一直线上的两个点之间的最大距离，超过此距离则被认为是两条不同的直线。

# HoughLinesP函数返回一个包含直线的列表，每条直线由其在图像中的起点和终点坐标表示。

# 直线检测
# !!!调整参数
show(edges, 'e')
tt = 60
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=60, maxLineGap=10)



# 直线分组
groups = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    group_added = False
    for group in groups:
        gx1, gy1, gx2, gy2 = group[0]
        if abs(y1 - gy1) < 10 and abs(y2 - gy2) < 10:
            group.append(line[0])
            group_added = True
            break
    if not group_added:
        groups.append([line[0]])

# 绘制直线
for group in groups:
    for line in group:
        x1, y1, x2, y2 = line
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 显示结果
show(image)







