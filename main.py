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
        
        image, PointsList =m.facePoint(gray, frame, data)
        eyesPoints= PointsList[36:42]
        rEyePoints = PointsList[42:48]
        RightEye =m.blinkDetector(PointsList[42:48])
        LeftEye=m.blinkDetector(eyesPoints)
        BlinkRatio = (RightEye+LeftEye)/2
        # print(BlinkRatio)
        cv.circle(frame, (40, 40), int(BlinkRatio*3), m.LIGHT_BLUE, 3)
        # cv.line(frame,(40, 70), (40,int(70-(BlinkRatio*6))),m.ORANGE,10)
        if BlinkRatio>=4:
            cv.putText(frame, f'Blink',(50, 70), cv.FONT_HERSHEY_COMPLEX, 0.9, m.PINK, 2 )
        # cv.circle(frame, rEyePoints[0], 3, m.GREEN, 2)
        # cv.circle(frame, rEyePoints[1], 3, m.YELLOW, 2)
        # 
        # cv.circle(frame, rEyePoints[3], 3, m.RED, 2)
        # cv.circle(frame, rEyePoints[4], 4, m.BLUE, 2)
        # cv.circle(frame, rEyePoints[5], 4, m.LIGHT_BLUE, 2)
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