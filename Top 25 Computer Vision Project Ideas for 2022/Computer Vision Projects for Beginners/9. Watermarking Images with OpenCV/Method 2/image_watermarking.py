import cv2
import numpy as np
from watermarking import watermarking

image = cv2.imread("bird nest.jpg")
watermark = cv2.imread("watermark3.png", cv2.IMREAD_UNCHANGED)
# Showing the result
final = watermarking(image, watermark)
cv2.imshow("Watermarked image",final)
cv2.imwrite("watermarked_bird_nest1.jpg", final)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Code Reference: https://viblo.asia/p/opencv-watermarking-image-1VgZv4or5Aw
