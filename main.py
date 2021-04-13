import cv2 as cv 
import numpy as np
import Module as m
import time
import dlib

# variables
fonts = cv.FONT_HERSHEY_PLAIN
frameCounter = 0
capID=0 
# objects  
camera = cv.VideoCapture(capID)
# staring time 
staringTimer = time.time()
while True:
    frameCounter += 1
    ret, frame= camera.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # cv.imshow("image", frame)
    image,data =m.faceDetector(frame, Draw=True)
    if data is not None:
        # m.BlinkingDetection(gray, data)
        # print('detected')
        
        image, PointLis =m.facePoint(gray, frame, data)
        eyesPoints= PointLis[36:42]
        BlinkRatio=m.blinkDetector(eyesPoints)
        # print(BlinkRatio)
        if BlinkRatio<=3:
            print("no Blink")
        elif BlinkRatio>3:
            print("Eyes Blinked")
        # cv.circle(frame, bPoints[0], 3, m.GREEN, 2)
        # cv.circle(frame, topMid, 3, m.YELLOW, 2)
        # cv.circle(frame, bottomMid, 3, m.CYAN, 2)
        # cv.circle(frame, bPoints[1], 3, m.RED, 2)
        # cv.circle(frame, eyesPoints[0], 4, m.BLUE, 2)
        # cv.circle(frame, eyesPoints[3], 4, m.BLUE, 2)
        # for center in eyesPoints:
            # cv.circle(frame, center,3, m.ORANGE, 1)
        # cv.circle(frame, PointLis[0], 2,m.LIGHT_BLUE, 1)

    seconds = time.time() - staringTimer
    # print(seconds)
    fps = frameCounter/seconds
    # print(fps)
    cv.putText(frame, f'FPS: {round(fps, 2)}',(20,20),fonts, 1.5, m.MAGENTA,2 )
    cv.imshow("image", image)
    key =cv.waitKey(1)
    if key ==ord('q'):
        break
cv.destroyAllWindows()
camera.release()