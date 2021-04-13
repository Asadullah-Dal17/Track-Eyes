import cv2 as cv 
import numpy as np
import dlib
import Module as m 
import time

# variables
fonts = cv.FONT_HERSHEY_PLAIN
frameCounter = 0
capID=1 
# objects  
camera = cv.VideoCapture(capID)
staringTimer = time.time()
while True:
    frameCounter += 1
    ret, frame= camera.read()
    cv.imshow("image", frame)
    image,ids, pts1, pts2, =m.faceDetector(frame, Draw=True)
    # print (ids,pts1, pts2)
    seconds = time.time() - staringTimer
    # print(seconds)
    fps = frameCounter/seconds
    # print(fps)
    cv.putText(frame, f'FPS: {round(fps, 2)}',(20,20),fonts, 2, (0,244,0),2 )
    cv.imshow("image", image)
    key =cv.waitKey(1)
    if key ==ord('q'):
        break
cv.destroyAllWindows()
camera.release()