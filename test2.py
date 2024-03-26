import cv2
import pytesseract
import matplotlib.pyplot as plt
import numpy as np

def show(image):
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

imagepath = "t1.jpg"



    
image =  cv2.imread(imagepath)

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)




# 进行图像增强，如去噪、增强对比度等
# 可以使用各种图像处理技术，如滤波、直方图均衡化等
# 这取决于你的具体需求和图像的特征

# 例如，可以使用高斯模糊去除噪声
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)

show(gray) 

# 使用自适应阈值二值化图像
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# show(thresh)

# 设置Tesseract OCR的路径
pytesseract.pytesseract.tesseract_cmd = r'D:\TOCR\tesseract'

# 进行OCR文本识别
text = pytesseract.image_to_string(gray, lang='chi_sim')
print(text)