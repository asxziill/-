# 上24
# 下24

import testFUN as fun

import cv2
from matplotlib import pyplot as plt
import os
import numpy as np





def testF(path):
    print(path)

    origin_image = cv2.imread(path)

    if origin_image is None:
        print("ERR")
        return "图像无法加载，请检查路径是否正确。"

    # print("原图")
    # fun.plt_show(origin_image)


    image = origin_image.copy()

    

    image = fun.getRectimage(image)

    return image

pgood = './good/IC_1.png'
    
p1 = './NG/IC_1.png'
p2 = './NG/IC_31.png'
p3 = './NG/IC_53.png'
p4 = './NG/IC_36.png'
p5 = './NG/IC_39.png'
pp = './NG/IC_27.png'


def test():
    path = './CQI-P/IC_'
    suf = '.png'
    for i in range(1, 10):
        impath = path + str(i) + suf
        testF(impath)

def testERR():
    path = './CQI-P/IC_'
    suf = '.png'
    for i in range(41, 50):
        impath = path + str(i) + suf
        testF(impath)

def testNG():
    path = './NG/IC_'
    suf = '.png'
    # for i in range(1, 3):
    #     impath = path + str(i) + suf
    #     testF(impath)

    for i in range(31, 40):
        impath = path + str(i) + suf
        testF(impath)

# tmp()

# test()
# testERR()
# testNG()




        
# img1 = testF(pgood)
img2 = testF(p2)

def tt(image, ksize=3, scale=1, delta=0):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Sobel operations
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize, scale=scale, delta=delta)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize, scale=scale, delta=delta)

    # Combining the gradients
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    combined = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    # Thresholding to get binary image
    _, binary = cv2.threshold(combined, 50, 255, cv2.THRESH_BINARY)

    fun.plt_show(image)
    fun.plt_show(binary)
    return binary



# tt(img1)

tt(img2)



# cv2.imwrite("gpttest2.png", img1)

# fun.detect_scratches(p1)

# import getText as tt

# it = tt.calculate_image_difference(img2)

# binary_image = cv2.inRange(it, 37, 39)

# fun.plt_show(it)
# fun.plt_show(binary_image)


# fun.plt_show(img1)
# fun.plt_show(img2)

# cv2.imwrite("gpttest1.png", img2)





# overlap = cv2.bitwise_and(img1, img2)
# fun.plt_show(overlap)

# # img1 = testF(path)
# # img2 = testF(errpath)





                            
