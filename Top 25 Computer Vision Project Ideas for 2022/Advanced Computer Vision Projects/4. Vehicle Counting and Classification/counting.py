import cv2
import numpy as np
from time import sleep

width_min=80 # Minimum rectangle width
height_min=80 # Minimum height of rectangle

offset=6 # Allowed error between pixels 

pos_line =550 # Count line position

delay= 60 # Video fps

detect = []
cars = 0

	
def handle_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

cap = cv2.VideoCapture('video.mp4')
subtract = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret , frame1 = cap.read()
    temp = float(1/delay)
    sleep(temp) 
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = subtract.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    expanded = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , kernel)
    expanded = cv2.morphologyEx (expanded, cv2. MORPH_CLOSE , kernel)
    outline,h=cv2.findContours(expanded,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame1, (25, pos_line), (1200, pos_line), (255,127,0), 3) 
    for(i,c) in enumerate(outline):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_outline = (w >= width_min) and (h >= height_min)
        if not validate_outline:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)        
        center = handle_center(x, y, w, h)
        detect.append(center)
        cv2.circle(frame1, center, 4, (0, 0,255), -1)

        for (x,y) in detect:
            if y < (pos_line+offset) and y > (pos_line - offset):
                cars += 1
                cv2.line(frame1, (25, pos_line), (1200, pos_line), (0,127,255), 3)  
                detect.remove((x,y))
                print("car is detected : "+str(cars))        
       
    cv2.putText(frame1, "VEHICLE COUNT : "+str(cars), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
    cv2.imshow("Video Original" , frame1)
    cv2.imshow("Detected",expanded)

    if cv2.waitKey(1) == ord('q'):
        break
    
cv2.destroyAllWindows()
cap.release()

# Code Reference- https://morioh.com/p/dd22880e3d8c