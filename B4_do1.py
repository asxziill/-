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
    # print("宽："+ str(width) + "  高 " + str(height))
    return rect

# 得到固定大小的芯片图像
def getRectimage(image, width = 300, height = 210):

    soruce_img = image.copy()

    # 先得到边缘图像
    image = getSide(image)

    # 对边缘进行形态学操作
    image = getRange(image)
    # 获得大矩形的信息
    rect = getChipRec(image)

    # 矩形左上角坐标往左上角微调
    y = rect[1] - 5
    x = rect[0] - 5
    resimage = soruce_img[y:y + height, x:x + width]

    return resimage

# 根据图像的灰度二值化图像
def getBinimage(image):
    gray_image = gray_guss(image)
    # 经过测试，189作为阈值可以区分开针脚于背景
    ret, thresh = cv2.threshold(gray_image, 189, 255, cv2.THRESH_BINARY)
    return thresh

# 利用形态学操作补充二值化去掉的阴影（利用了垂直矩形) (传入核的大小填充阴影)
def getRange_size(image, width, height):
    kernel_size = (width, height)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # 应用垂直闭运算
    closing_vertical = cv2.morphologyEx(image, cv2.MORPH_CLOSE, vertical_kernel)

    return closing_vertical

# 移除高度较小的连通块
def remove_bad(image, height_threshold = 5):
    # 查找轮廓
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个全黑的图像用于绘制保留的轮廓
    mask = np.zeros_like(image)

    # 遍历轮廓，移除高度小于阈值的连通块
    for contour in contours:
        _, _, _, h = cv2.boundingRect(contour)  # 获取轮廓的边界矩形
        if h >= height_threshold:  # 保留高度大于等于阈值的轮廓
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    return mask

# 得到针脚的二值化图像
def getBinRect(image):
    image = getBinimage(image)
    # 先处理阴影，保证一个针脚对应一个矩形
    image = getRange_size(image, 1, 10)
    # 移除断掉的针脚
    image = remove_bad(image)

    return image

# 得到图片的针脚的二值化图像（已截图
def get(path):
    origin_image = cv2.imread(path)

    if origin_image is None:
        return "图像无法加载，请检查路径是否正确。"

    image = getRectimage(origin_image)

    image = getBinRect(image)

    return image


# 上下排经过测试有150像素为分界点(返回带编号的针脚矩形（从左到右)
def get_expanded_rectangles(binary_image, y_threshold = 150):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    expanded_rects = [cv2.boundingRect(contour) for contour in contours]

    # 根据y坐标和x坐标排序
    def sort_key(rect):
        y_priority = 0 if rect[1] < y_threshold else 1
        return (y_priority, rect[0])

    expanded_rects.sort(key=sort_key)

    # 为每个矩形分配编号
    numbered_rects = [(i + 1, rect) for i, rect in enumerate(expanded_rects)]
    
    return numbered_rects

# 得到正常的图像
def Get():
    path = './CQI-P/IC_'
    suf = '.png'

    # 利用前4张图像生成正常图像的针脚二值图像
    overlap = get(path + str(1) + suf)

    for i in range(2, 4):
        impath = path + str(i) + suf
        img = get(impath)

        overlap = cv2.bitwise_and(overlap, img)

    # 形态学操作
    overlap = getRange_size(overlap, 1, 10)

    return get_expanded_rectangles(overlap)

numAndRect = Get()

# 获得图像缺失针脚的编号
def check_overlap(defect_binary, numbered_rects, overlap_threshold=0.1):
    contours, _ = cv2.findContours(defect_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个包含所有normal_rects编号的集合
    remaining_numbers = set(number for number, _ in numbered_rects)

    for contour in contours:
        defect_rect = cv2.boundingRect(contour)
        defect_area = defect_rect[2] * defect_rect[3]

        for number, normal_rect in numbered_rects:
            # 计算重叠区域
            dx = min(defect_rect[0] + defect_rect[2], normal_rect[0] + normal_rect[2]) - max(defect_rect[0], normal_rect[0])
            dy = min(defect_rect[1] + defect_rect[3], normal_rect[1] + normal_rect[3]) - max(defect_rect[1], normal_rect[1])
            if dx > 0 and dy > 0:
                overlap_area = dx * dy

                # 检查重叠比例
                if overlap_area / defect_area > overlap_threshold:
                    # 如果找到匹配的重叠矩形，从集合中移除对应的编号
                    remaining_numbers.discard(number)
                    break  # 假设一个连通块只能匹配一个矩形

    return remaining_numbers

def format_missing_pins_message(remaining_numbers):
    # 检查是否有剩余的编号
    if not remaining_numbers:
        return "没有引脚缺失"

    # 分离上排和下排的引脚编号
    upper_row_numbers = [num for num in remaining_numbers if num <= 24]
    lower_row_numbers = [num - 24 for num in remaining_numbers if num > 24]

    # 格式化消息
    messages = []
    if upper_row_numbers:
        upper_row_str = ', '.join(str(num) for num in sorted(upper_row_numbers))
        messages.append('上排第 ' + upper_row_str + ' 根引脚缺失')
    
    if lower_row_numbers:
        lower_row_str = ', '.join(str(num) for num in sorted(lower_row_numbers))
        messages.append('下排第 ' + lower_row_str + ' 根引脚缺失')

    return '\n'.join(messages)

def identify_missing_pins(image_path):
    # 读取并处理图像
    origin_image = cv2.imread(image_path)
    if origin_image is None:
        return "图像无法加载，请检查路径是否正确。"

    gray = gray_guss(origin_image)
    _, otsu_thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt_show(otsu_thresholded)
    
    print(image_path)
     # 截图
    image = getRectimage(origin_image)
    plt_show(origin_image)

    # 针脚二值化
    image = getBinRect(image)

    # 获得该图像未出现过的进行编号
    remaining_numbers = check_overlap(image, numAndRect)

    outmessage = format_missing_pins_message(remaining_numbers)
    print(outmessage)

# 输入图像编号左端点和右端点
def work(l, r):
    path = './CQI-P/IC_'
    suf = '.png'
    for i in range(l, r):
        impath = path + str(i) + suf
        identify_missing_pins(impath)

# work(1, 10)
work(23, 24)