import cv2
from matplotlib import pyplot as plt
import os
import numpy as np

# plt显示灰度图片
def plt_show(img):
    plt.imshow(img,cmap='gray')
    plt.show()

# 图像去噪灰度处理 返回灰度图像
def gray_guss(image):

    image = cv2.GaussianBlur(image, (3, 3), 0)

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_image

# 根据图像的灰度二值化图像
def getBinimage(image, lim = 189):
    gray_image = gray_guss(image)
    # 经过测试，189作为阈值可以区分开针脚于背景
    ret, thresh = cv2.threshold(gray_image, lim, 255, cv2.THRESH_BINARY)
    return thresh

def remove_small_area_contours(image, area_threshold = 200):
    # 查找轮廓
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个全黑的图像用于绘制保留的轮廓
    mask = np.zeros_like(image)

    # 遍历轮廓，移除面积小于阈值的连通块
    for contour in contours:
        area = cv2.contourArea(contour)  # 获取轮廓的面积
        if area >= area_threshold:  # 保留面积大于等于阈值的轮廓
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    return mask

# 利用形态学操作补充二值化去掉的阴影（利用了垂直矩形) (传入核的大小填充阴影)
def getRange_size(image, width, height):
    kernel_size = (width, height)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # 应用垂直闭运算
    closing_vertical = cv2.morphologyEx(image, cv2.MORPH_CLOSE, vertical_kernel)

    return closing_vertical

# 得到针脚二值图像的大矩形
def getBinBigRect(image):
    image = getBinimage(image)
    # 封闭针脚的阴影
    image = getRange_size(image, 10, 10)
    image = remove_small_area_contours(image, 650)
    # plt_show(image)
    return image

# 旋转芯片
def RotateImg(source_img):
    
    image = source_img.copy()
    
    # 先得到边缘图像
    Bin_image = getBinBigRect(image)

    # 查找轮廓
    contours, _ = cv2.findContours(Bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 计算所有轮廓的平均倾斜角度
    angles = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        (x, y), (width, height), angle = rect

        # 如果宽度小于高度，调整角度
        if width < height:
            angle = 90 + angle

        
        angles.append(angle)

    if not angles:
        return None  # 如果没有找到轮廓，则返回None

    avg_angle = np.mean(angles)

    if avg_angle >= 180:
            avg_angle = avg_angle - 180


    # 旋转图像以校正倾斜
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated

# 图片获得边缘
def getSide(image):
    # 图像去噪灰度处理
    gray_image = gray_guss(image)

    Sobel_x = cv2.Sobel(gray_image, cv2.CV_16S, 1, 0)

    absX = cv2.convertScaleAbs(Sobel_x)
    
    image = absX

    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh
    # 轮廓处理完

# 形态学操作，主要闭x轴(传入二值图像)
def getRange(image):

    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))

    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX,iterations = 1)

    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))

    image = cv2.dilate(image, kernelX)
    image = cv2.erode(image, kernelX)

    image = cv2.erode(image, kernelY)
    image = cv2.dilate(image, kernelY)

    image = cv2.medianBlur(image, 21)

    return image

# 得到芯片矩形信息 范围一个包住所有轮廓的矩形
def getChipRec(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    list_x = []
    list_y = []
    list_rx = []
    list_ry = []
    for item in contours:
        rect = cv2.boundingRect(item)
        list_x.append(rect[0])
        list_y.append(rect[1])
        list_rx.append(rect[0] + rect[2])
        list_ry.append(rect[1] + rect[3])
        
    # x,y取最小值
    # 右下取最大值
    x = min(list_x)
    y = min(list_y)

    rx = max(list_rx)
    ry = max(list_ry)

    width = rx - x
    height = ry - y
    rect = []
    rect.append(x)
    rect.append(y)
    rect.append(width)
    rect.append(height)
    return rect

# 得到芯片矩形的图像(宽和高设为定值，保证图片大小一致)
def getRectimage(image, width = 300, height = 120):

    image = RotateImg(image)

    # plt_show(image)
    soruce_img = image.copy()

    # 先得到边缘图像
    image = getSide(image)
    # plt_show(image)

    # 对边缘进行形态学操作
    image = getRange(image)
    image = remove_small_area_contours(image, 650)
    # plt_show(image)
    
    # 获得大矩形的信息
    rect = getChipRec(image)
    

    # 矩形左上角坐标往左上角微调
    y = rect[1] + 45
    x = rect[0] - 5
    resimage = soruce_img[y:y + height, x:x + width]


    return resimage

def equalized(image):
    gray_image = gray_guss(image)
    equalized_image = cv2.equalizeHist(gray_image)
    return equalized_image

def detect_and_draw_lines(binary_image, original_image, length_threshold):
    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(binary_image, 1, np.pi/180, 100, minLineLength=length_threshold, maxLineGap=2)

    # Draw lines on the original image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(original_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color
    else:
        print("未检测到划痕，产品合格")

    return original_image


def identify_scratch_marks(image_path):
    # 读取并处理图像
    print(image_path)
    origin_image = cv2.imread(image_path)
    if origin_image is None:
        return "图像无法加载，请检查路径是否正确。"
    
    plt_show(origin_image)
    image = origin_image.copy()
    
    image = getRectimage(image)
    plt_show(image)

    equalized_image = equalized(image)

    Bin_image = cv2.inRange(equalized_image, 220, 255)

    identify_image = detect_and_draw_lines(Bin_image, image, 10)

    plt_show(identify_image)

    

# 输入图像编号左端点和右端点
def work(num, File):
    path = './' + File + '/IC_'
    suf = '.png'

    print("\r")

    if File == "good":
        print("未检测到划痕，产品合格")
        return
    
    impath = path + str(num) + suf

    identify_scratch_marks(impath)




work(26, "NG")
work(46, "NG")
work(49, "NG")
work(50, "NG")
work(51, "NG")



