import cv2 as cv 
import numpy as np
import dlib

# variables
capID=1 
# objects  
camera = cv.VideoCapture(capID)

while True:
    ret, frame= camera.read()
    cv.imshow("image", frame)
    key =cv.waitKey(1)
    if key ==ord('q'):
        break
cv.destroyAllWindows()
camera.release()