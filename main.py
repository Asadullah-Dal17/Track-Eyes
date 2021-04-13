import cv2 as cv 
import numpy as np
import Module as m 
import time
import dlib

# variables
fonts = cv.FONT_HERSHEY_PLAIN
frameCounter = 0
capID=1 
# objects  
camera = cv.VideoCapture(capID)
# predictor = dlib.shape_predictor('Predictor/shape_predictor_68_face_landmarks.dat')
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
        cv.circle(frame, PointLis[0], 3,m.LIGHT_BLUE, 2)
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