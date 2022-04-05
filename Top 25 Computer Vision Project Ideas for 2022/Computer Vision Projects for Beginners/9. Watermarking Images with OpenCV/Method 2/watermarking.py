import cv2
import numpy as np

def watermarking(original, watermark, alpha = 0.5, width=500, height=500):
  # resize image
#  original = cv2.resize(original, (width, height), interpolation = cv2.INTER_AREA)
#  originalHeight, originalWidth,_ = original.shape
#  original = np.dstack([original, np.ones((originalHeight,originalWidth), dtype="uint8") * 255])
  
  
  #Resizing the watermark image
    scale = 40
    rw = int(watermark.shape[1] * scale / 100)
    rh = int(watermark.shape[0] * scale / 100)
    dim = (rw,rh)
    watermarked = cv2.resize(watermark, dim, interpolation = cv2.INTER_AREA)
    wH, wW,_ = watermarked.shape
    
      
    # resize image
#    percent_of_scaling = 20
#    new_width = int(original.shape[1] * percent_of_scaling/100)
#    new_height = int(original.shape[0] * percent_of_scaling/100)
    new_dim = (width, height)
    resized_img = cv2.resize(original, new_dim, interpolation=cv2.INTER_AREA)
    h_img, w_img, _ = resized_img.shape



    #Blending
    #overlay = np.zeros((originalHeight, originalWidth, 4), dtype="uint8")
    #overlay[10:10 + wH, 10:10 + wW] = watermarked
    top_y = int(wH/2)
    left_x = int(wW/2)
      
    bottom_y = top_y + wH
    right_x = left_x + wW

    roi = resized_img[top_y:bottom_y, left_x:right_x]
    
  
    result= cv2.addWeighted(roi,1,watermarked,0.4,0)
    resized_img[top_y:bottom_y, left_x:right_x] = result
    
    return resized_img

# Code Reference: 
# 1. https://viblo.asia/p/opencv-watermarking-image-1VgZv4or5Aw
# 2. https://github.com/harika-bonthu/Watermark-OpenCV/blob/main/watermark.ipynb
