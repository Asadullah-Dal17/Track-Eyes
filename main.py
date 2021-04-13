import cv2 as cv 
import numpy as np
import dlib
import Module as m 
# variables
capID=1 
# objects  
camera = cv.VideoCapture(capID)

while True:
    ret, frame= camera.read()
    cv.imshow("image", frame)
    image =m.faceDetector(frame, Draw=True)
    cv.imshow("image", image)
    key =cv.waitKey(1)
    if key ==ord('q'):
        break
cv.destroyAllWindows()
camera.release()