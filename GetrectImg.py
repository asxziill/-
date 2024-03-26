import cv2
import numpy as np

def process_image(image_path):
    # Load the image
    img = cv2.imread(image_path)
    assert img is not None

    # Resize the image
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert gray is not None

    # Apply Gaussian blur
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), -1)

    # Apply mean blur
    mean_blurred = cv2.blur(gray_blurred, (15, 15))
    assert mean_blurred is not None

    # Calculate the absolute difference
    diff = cv2.absdiff(mean_blurred, gray_blurred)
    assert diff is not None

    # Apply threshold
    _, thresh = cv2.threshold(diff, 5, 255, cv2.THRESH_BINARY)
    assert thresh is not None

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Draw contours
    img_copy = img.copy()
    for contour in contours:
        rr = cv2.minAreaRect(contour)
        max_side = max(rr[1])
        min_side = min(rr[1])

        if rr[1][0] * rr[1][1] > 1000 and rr[1][0] * rr[1][1] < 1000000 and \
           max_side / min_side > 3 and max_side / min_side < 4:
            cv2.drawContours(img_copy, [contour], -1, (0, 0, 255))

    # Display the images
    cv2.imshow('Difference', diff)
    cv2.imshow('Threshold', thresh)
    cv2.imshow('Contours', img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Replace 'path_to_image' with your actual image path
process_image('gpttest2.png')
