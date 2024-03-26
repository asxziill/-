from PIL import Image
import pytesseract
import matplotlib.pyplot as plt  
# %matplotlib inline

path="E:\大三上\数图\大作业参考\\1.jpg"
# path="./mat.png"

"""
🐬指明tesseract命令位置
"""

tesseract_cmd = r'D:\TOCR\tesseract'
pytesseract.pytesseract.tesseract_cmd =tesseract_cmd



# 显示
image=Image.open(path)
plt.figure(figsize=(2,2))
plt.axis('off')
plt.imshow(image)



print(123)
print(pytesseract.image_to_string(image, lang='chi_sim'))

