import cv2
import numpy as np

path = "t1.jpg"
# 读取图像
img = cv2.imread(path)

# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 应用 Canny 边缘检测算法
edges = cv2.Canny(gray, 50, 150)

# 找到轮廓
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 遍历所有轮廓
for cnt in contours:
    # 对于每个轮廓，检测是否是矩形
    if len(cnt) >= 4:
        # 计算轮廓的外接矩形
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 绘制外接矩形
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

# 显示图像
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()