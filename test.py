import imutils
import cv2
from imutils.video import VideoStream
import time

camera = VideoStream(src=0).start()
time.sleep(2)

while True:
    frame = camera.read()
    cv2.imshow("YOLO", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
