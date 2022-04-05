from imutils.video import VideoStream
import imutils
import time
import cv2
from watermarking import watermarking

print("[INFO] camera sensor warming up...")
vs = VideoStream(0).start()
time.sleep(2.0)
watermark = cv2.imread("watermark2.png", cv2.IMREAD_UNCHANGED)
# loop over the frames from the video stream
while True:
      # grab the frame from the threaded video stream, resize it to
      # have a maximum width of 400 pixels, and convert it to
      # grayscale
    frame = vs.read()
    frame = imutils.resize(frame, height=500)
    h,w, = frame.shape[:2]

    # show the frame
    final = watermarking(frame, watermark, width=w, height=h)
    cv2.imshow("Original", frame)
    cv2.imshow("Watermarked_", final)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
vs.stop()

cv2.destroyAllWindows()

# Code Reference: https://viblo.asia/p/opencv-watermarking-image-1VgZv4or5Aw
