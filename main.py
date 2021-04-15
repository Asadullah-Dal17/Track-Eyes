import cv2 as cv
import numpy as np
import Module as m
import time
import dlib

# variables
fonts = cv.FONT_HERSHEY_PLAIN
frameCounter = 0
capID = 0
# objects
camera = cv.VideoCapture(capID)
# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
Recoder = cv.VideoWriter('output.mp4', fourcc, 15.0, (640, 480))

# staring time
staringTimer = time.time()
while True:
    frameCounter += 1
    ret, frame = camera.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # cv.imshow("image", frame)
    image, data = m.faceDetector(frame, Draw=True)
    if data is not None:
        # m.BlinkingDetection(gray, data)
        # print('detected')
        image, PointsList = m.facePoint(gray, frame, data)
        RightEyeList = PointsList[36:42]
        LeftEyeList = PointsList[42:48]
        # ArryN = np.array(RightEyeList, dtype=np.int32)
        # print(ArryN)
        # mask = np.zeros(frame.shape, dtype=np.uint8)
        # cv.fillPoly(mask, [ArryN], (0,255,255))
        RightEye = m.blinkDetector(PointsList[42:48])
        mask, RightCropd, EyePosition = m.EyesTracking(frame, RightEyeList)
        mask2, LeftCroped, LeftEyePos = m.EyesTracking(frame, LeftEyeList)
        CurrentPos, color = m.Position(LeftEyePos)
        indictor = np.zeros((200, 200, 3), dtype=np.uint8)
        indictor[:] = color[0]
        cv.line(frame, (30, 90), (120, 90), color[1], 40)
        if EyePosition is not None:
            print(CurrentPos)
            cv.putText(frame, f"P: {CurrentPos}",
                       (35, 95), m.fonts, 0.6, color[0], 2)
            # # print(EyePosition, EyePosition.index(max(EyePosition)))
            # if EyePosition.index(max(EyePosition)) == 0:
            #     print("Right")
            # elif EyePosition.index(max(EyePosition)) == 1:
            #     print("Center")
            # elif EyePosition.index(max(EyePosition)) == 2:
            #     print('Left')

        if mask2 is not None:
            # cv.imshow('mask2', mask2)
            # cv.imshow("mask", mask)
            cv.imshow('Right', RightCropd)
        LeftEye = m.blinkDetector(LeftEyeList)
        BlinkRatio = (RightEye+LeftEye)/2
        # print(BlinkRatio)
        cv.circle(frame, (40, 40), int(BlinkRatio*3), m.LIGHT_BLUE, 3)
        # cv.line(frame,(40, 70), (40,int(70-(BlinkRatio*6))),m.ORANGE,10)
        if BlinkRatio >= 4:
            cv.putText(frame, f'Blink', (50, 70),
                       cv.FONT_HERSHEY_COMPLEX, 0.9, m.PINK, 2)
        # cv.circle(frame, RightEyeList[0], 3, m.GREEN, 2)
        # cv.circle(frame, RightEyeList[1], 3, m.YELLOW, 2)
        #
        # cv.circle(frame, RightEyeList[3], 3, m.RED, 2)
        # cv.circle(frame, RightEyeList[4], 4, m.BLUE, 2)
        # cv.circle(frame, RightEyeList[5], 4, m.LIGHT_BLUE, 2)
        # for center in LeftEyeList:
            # cv.circle(frame, center,3, m.ORANGE, 1)
        # cv.circle(frame, PointLis[0], 2,m.LIGHT_BLUE, 1)

    seconds = time.time() - staringTimer
    # print(seconds)
    fps = frameCounter/seconds
    # print(fps)
    cv.putText(frame, f'FPS: {round(fps, 2)}',
               (20, 20), fonts, 1.5, m.MAGENTA, 2)
    cv.imshow("image", image)
    # cv.imshow('Indicator', indictor)
    Recoder.write(frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break
cv.destroyAllWindows()
Recoder.release()
camera.release()
