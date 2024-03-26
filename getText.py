

import cv2
import numpy as np

def equalize_histogram(image):
    # Check if the image is loaded correctly
    if image is None:
        return None, "Error: Image could not be loaded."
    
    # Convert to grayscale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(image)
    
    # return equalized_image, None
    return equalized_image

def calculate_image_difference(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image2 = cv2.equalizeHist(image1)
    
    # Calculate the absolute difference between the images
    difference_image = cv2.absdiff(image1, image2)
    
    return difference_image

# 使用示例
# img = cv2.imread('your_image_path.png')
# result, error = equalize_histogram(img)
# if error is None:
#     cv2.imshow('Histogram Equalized Image', result)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print(error)


