import cv2
import numpy as np

# 展示图像
def cv_show(self,name,img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#  灰度转化
def convert_gray_scale(self,image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# 二值化图像，将灰度第或高滤去
def select_rgb_white_yellow(self,image): 
    #过滤掉背景
    lower = np.uint8([120, 120, 120])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    self.cv_show('white_mask',white_mask)
    
    masked = cv2.bitwise_and(image, image, mask = white_mask)
    self.cv_show('masked',masked)
    return masked

# 边缘检测的结果
def detect_edges(self,image, low_threshold=50, high_threshold=200):
    return cv2.Canny(image, low_threshold, high_threshold)

# 提取区域 根据项目
def select_region(self,image):
    rows, cols = image.shape[:2]
    pt_1  = [cols*0.05, rows*0.90]
    pt_2 = [cols*0.05, rows*0.70]
    pt_3 = [cols*0.30, rows*0.55]
    pt_4 = [cols*0.6, rows*0.15]
    pt_5 = [cols*0.90, rows*0.15] 
    pt_6 = [cols*0.90, rows*0.90]

    vertices = np.array([[pt_1, pt_2, pt_3, pt_4, pt_5, pt_6]], dtype=np.int32) 
    point_img = image.copy()       
    point_img = cv2.cvtColor(point_img, cv2.COLOR_GRAY2RGB)
    for point in vertices[0]:
        cv2.circle(point_img, (point[0],point[1]), 10, (0,0,255), 4)
    self.cv_show('point_img',point_img)       
    return self.filter_region(image, vertices)

# 直线检测
def hough_lines(self,image):       
    return cv2.HoughLinesP(image, rho=0.1, theta=np.pi/10, threshold=15, minLineLength=9, maxLineGap=4)