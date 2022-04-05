# import the necessary packages
from pyimagesearch.face_blurring import anonymize_face_pixelate
from pyimagesearch.face_blurring import anonymize_face_simple
import numpy as np
import argparse
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-f", "--face", required=True,
	help="path to face detector model directory")
ap.add_argument("-m", "--method", type=str, default="simple",
	choices=["simple", "pixelated"],
	help="face blurring/anonymizing method")
ap.add_argument("-b", "--blocks", type=int, default=20,
	help="# of blocks for the pixelated blurring method")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000_fp16.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the input image from disk, clone it, and grab the image spatial
# dimensions
image = cv2.imread(args["image"])

# Scale the image to a smaller size for display
percent_of_scaling = 20
new_width = int(image.shape[1] * percent_of_scaling/100)
new_height = int(image.shape[0] * percent_of_scaling/100)
new_dim = (new_width, new_height)
resized_img = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)


orig = resized_img.copy()
h, w, _ = resized_img.shape

# construct a blob from the image
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),(104.0, 177.0, 123.0))

# pass the blob through the network and obtain the face detections
print("[INFO] computing face detections...")
net.setInput(blob)
detections = net.forward()


# loop over the detections
for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# detection
 
	confidence = detections[0, 0, i, 2]
	# filter out weak detections by ensuring the confidence is greater
	# than the minimum confidence
 
	if confidence > args["confidence"]:
		# compute the (x, y)-coordinates of the bounding box for the
		# object
  
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
  
		# extract the face ROI
		face = resized_img[startY:endY, startX:endX]
  
  
  		# check to see if we are applying the "simple" face blurring
		# method
		if args["method"] == "simple":
			face = anonymize_face_simple(face, factor=3.0)
   
		# otherwise, we must be applying the "pixelated" face
		# anonymization method
		else:
			face = anonymize_face_pixelate(face,
				blocks=args["blocks"])
   
		# store the blurred face in the output image
		resized_img[startY:endY, startX:endX] = face
  
# display the original image and the output image with the blurred
# face(s) side by side
output = np.hstack([orig, resized_img])
cv2.imshow("Output", output)
cv2.waitKey(0)



# Run the following command in terminal to display the output image:
# python blur_face.py --image examples/example1.jpg --face face_detector


# Code Refrence: https://pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python/


# To download 'deploy.prototxt' and 'res10_300x300_ssd_iter_140000_fp16.caffemodel' files:
# https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
# https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel