import cv2
from matplotlib import pyplot as plt
import os
import numpy as np

# plt显示彩色图片
def plt_show0(img):
#cv2与plt的图像通道不同：cv2为[b,g,r];plt为[r, g, b]
    b,g,r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.show()

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

    # x方向上的边缘检测（增强边缘信息）

    # gradient_x = cv2.Sobel(src, ddepth, dx, dy, ksize)
    # src：输入的图像。
    # ddepth：输出图像的深度，通常设置为-1表示与输入图像相同的深度。
    # dx和dy：表示求导的阶数。dx=1表示计算水平方向的梯度，dy=1表示计算垂直方向的梯度，dx=0和dy=1表示只计算垂直方向的梯度，dx=1和dy=0表示只计算水平方向的梯度。
    # ksize：Sobel算子的核大小，通常设置为正奇数，例如3、5、7等。较大的核大小可以检测到较长的边缘，但也会导致边缘变得模糊
    # 函数会返回一个经过Sobel算子处理后的输出图像，其中每个像素的值表示该像素的梯度强度。
    Sobel_x = cv2.Sobel(gray_image, cv2.CV_16S, 1, 0)

    # output_image = cv2.convertScaleAbs(input_image, alpha, beta)
    # input_image：输入的图像。
    # alpha：尺度系数，用于控制线性变换的缩放比例。
    # beta：平移系数，用于控制线性变换的平移量。
    # 函数会返回一个经过线性变换和绝对值转换后的输出图像。
    absX = cv2.convertScaleAbs(Sobel_x)
    
    image = absX

    # 图像阈值化操作——获得二值化图
    # ret, thresholded_image = cv2.threshold(src, thresh, maxval, threshold_type)
    # src：输入的图像，通常为灰度图像。
    # thresh：阈值，用于将像素值与之比较。
    # maxval：最大像素值，用于指定分割后的像素值。
    # threshold_type：阈值处理的类型，可以使用OpenCV提供的常量来指定不同的处理方式，
    # 如cv2.THRESH_BINARY表示二值化处理，cv2.THRESH_BINARY_INV表示反二值化处理等。
    # 函数会返回一个二值图像，其中像素值根据阈值处理的结果设置为0或maxval。
    # ret  表示阈值处理的阈值
    # ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 显示灰度图像
    # plt_show(image)

    return thresh
    # 轮廓处理完


# 形态学操作，(传入二值图像)
def getRange(image):
    # 形态学（从图像中提取对表达和描绘区域形状有意义的图像分量）——闭操作

    # structuring_element = cv2.getStructuringElement(shape, ksize)
    # shape：结构元素的形状，可以使用OpenCV提供的常量来指定，
    # 如cv2.MORPH_RECT表示矩形形状，cv2.MORPH_ELLIPSE表示椭圆形状，cv2.MORPH_CROSS表示十字形状等。
    # ksize：结构元素的大小，通常是一个正整数，表示结构元素的宽度和高度。
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))

    # output_image = cv2.morphologyEx(input_image, op, kernel)
    # input_image：输入的图像。
    # op：形态学操作的类型，可以使用OpenCV提供的常量来指定不同的操作，
    # 如cv2.MORPH_ERODE表示腐蚀操作，cv2.MORPH_DILATE表示膨胀操作，cv2.MORPH_OPEN表示开运算，cv2.MORPH_CLOSE表示闭运算等。
    # kernel：结构元素（kernel），用于指定形态学操作的范围和形状。可以使用cv2.getStructuringElement()函数创建结构元素
    # iterations迭代次数
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX,iterations = 1)
    # plt_show(image)
    # 闭操作填充图像中的小孔洞
    # 显示灰度图像


    # 获得区间图像

    # 腐蚀（erode）和膨胀（dilate）
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))

    #x方向进行闭操作（抑制暗细节）
    # 膨胀后 腐蚀
    image = cv2.dilate(image, kernelX)
    image = cv2.erode(image, kernelX)

    #y方向的开操作
    image = cv2.erode(image, kernelY)
    image = cv2.dilate(image, kernelY)

    # 中值滤波（去噪）
    image = cv2.medianBlur(image, 21)

    # 此时图像已经区域锐化
    # 显示灰度图像
    # plt_show(image)
    return image



def debugcontours(contours, img):
    for item in contours:
        rect = cv2.boundingRect(item)
        x = rect[0]
        y = rect[1]
        weight = rect[2]
        height = rect[3]

        image = img[y:y + height, x:x + weight]
        plt_show(image)
    


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

# 得到针脚的矩形图像
def getBinRect(image):
    image = getBinimage(image)
    # 先处理阴影，保证一个针脚对应一个矩形
    image = getRange_size(image, 1, 10)
    # 移除断掉的针脚
    image = remove_bad(image)

    return image

# 统计针脚的数量，需要传入二值图像
def count_rectangles(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    num_rectangles = len(contours)

    return num_rectangles


# 上下排经过测试有150像素为分界点
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


def debugRect(size, rectangles):
    # 创建一张全黑的图像
    black_image = np.zeros(size, dtype=np.uint8)

    # 在黑色背景上绘制矩形
    for rect in rectangles:
        cv2.rectangle(black_image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), 255, 1)

    return black_image

def get_remaining_rectangles(numbered_rects, remaining_numbers):
    remaining_rects = []
    for number, rect in numbered_rects:
        if number in remaining_numbers:
            remaining_rects.append(rect)
    return remaining_rects

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
    

# 得到针脚二值图像的大矩形
def getBinBigRect(image):
    image = getBinimage(image)
    # 封闭针脚的阴影
    image = getRange_size(image, 8, 8)
    image = remove_small_area_contours(image, 50)
    return image

# 旋转芯片
def RotateImg(source_img):
    image = source_img.copy()

    # 先得到边缘图像
    Bin_image = getBinBigRect(image)

    # 对边缘进行形态学操作
    morph_image = getRange_size(Bin_image, 10, 10)
    morph_image = remove_small_area_contours(morph_image)


    # 查找轮廓
    contours, _ = cv2.findContours(morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 计算所有轮廓的平均倾斜角度
    angles = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        (x, y), (width, height), angle = rect

        # 如果宽度小于高度，调整角度
        if width < height:
            angle = 90 + angle

        if angle >= 180:
            angle = angle - 180

        angles.append(angle)

    if not angles:
        return None  # 如果没有找到轮廓，则返回None

    avg_angle = np.mean(angles)

    # 旋转图像以校正倾斜
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated


# 得到芯片矩形的图像(宽和高设为定值，保证图片大小一致)
def getRectimage(image, width = 300, height = 120, isrotate = True):

    if isrotate:
        image = RotateImg(image)

    plt_show(image)

    soruce_img = image.copy()

    # 先得到边缘图像
    image = getSide(image)

    # 对边缘进行形态学操作
    image = getRange(image)
    
    # 获得大矩形的信息
    rect = getChipRec(image)

    # 矩形左上角坐标往左上角微调
    y = rect[1] + 45
    x = rect[0] - 5
    resimage = soruce_img[y:y + height, x:x + width]


    return resimage

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


def detect_scratches(file_path):
    # Load the image
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, "Error: Image could not be loaded."
    
    # Apply Fourier Transform
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Create a bandpass filter
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow-30:crow+30, :] = 1  # Vertical band
    
    # Apply the mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    # Normalize the result
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)
    
    # Apply threshold to get binary image
    _, img_thresh = cv2.threshold(img_back, 50, 255, cv2.THRESH_BINARY)  # Threshold might need adjustment
    
    return img_thresh