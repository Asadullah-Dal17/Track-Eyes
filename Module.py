import cv2 as cv
import numpy as np
import dlib
import math
# variables
fonts = cv.FONT_HERSHEY_COMPLEX
# colors
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
GOLDEN = (32, 218, 165)
LIGHT_BLUE = (255, 30, 144)
PURPLE = (128, 0, 128)
CHOCOLATE = (30, 105, 210)
PINK = (147, 20, 255)
ORANGE = (0, 69, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
# objects
detectFace = dlib.get_frontal_face_detector()
# landmarks detector
predictor = dlib.shape_predictor(
    "Predictor/shape_predictor_68_face_landmarks.dat")

# function


def midpoint(pts1, pts2):
    x, y = pts1
    x1, y1 = pts2
    xOut = int((x + x1)/2)
    yOut = int((y1 + y)/2)
    # print(xOut, x, x1)
    return (xOut, yOut)


def eucaldainDistance(pts1, pts2):
    x, y = pts1
    x1, y1 = pts2
    eucaldainDist = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)

    return eucaldainDist


def faceDetector(image, Draw=True):
    pts1 = (0, 0)
    pts2 = (0, 0)
    IDs = 0
    Gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = detectFace(Gray)
    face = None
    for face in faces:
        # print(face)
        pts1 = (face.left(), face.top())
        pts2 = (face.right(), face.bottom())
        # landmarks = predictor(Gray, face)
        # x=landmarks.part(36).x
        # y=landmarks.part(36).y
        # print(type(landmarks))
        if Draw == True:
            cv.rectangle(image, pts1, pts2, (0, 255, 255), 1)
    # print(data)
    return image, face


def facePoint(grayImage, image, face, Draw=False):
    landmarks = predictor(grayImage, face,)
    pointsList = []
    for n in range(0, 68):
        point = (landmarks.part(n).x, landmarks.part(n).y)
        pointsList.append(point)
        if Draw == True:
            cv.circle(image, point, 2, ORANGE, 2)
    # print(pointsList)
    # print(point)
    return image, pointsList


def blinkDetector(eyePoints):
    # TODO find blinking points
    top = eyePoints[1:3]
    bottom = eyePoints[4:6]
    topMid = midpoint(top[0], top[1])
    bottomMid = midpoint(bottom[0], bottom[1])
    # print(topMid)
    Vertical = eucaldainDistance(topMid, bottomMid)
    Horizontal = eucaldainDistance(eyePoints[0], eyePoints[3])
    blinkRatio = (Horizontal/Vertical)

    # print(Vertical, '   ', Horizontal)
    # return top, bottom,topMid, bottomMid
    return blinkRatio


def EyesTracking(image, EyesPoints, Draw=True):
    EyePosition = "None"
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    dim = gray.shape
    mask = np.zeros(dim, dtype=np.uint8)
    # print('noting mask')
    # Drawing Right on the mask
    RightPolly = np.array(EyesPoints, dtype=np.int32)

    # Fill the Region of Eye on the mask
    cv.fillPoly(mask, [RightPolly], 255)

    # Extrating the Eyes from the image using and operation
    eye = cv.bitwise_and(gray, gray, mask=mask)
    # TODO Extract the eyes form frame,
    # find the mix and min x and y values of Eyes Positions
    # print(EyesPoints[0][0])
    maxX = (max(EyesPoints, key=lambda item: item[0]))[0]
    minX = (min(EyesPoints, key=lambda item: item[0]))[0]
    maxY = (max(EyesPoints, key=lambda item: item[1]))[1]
    minY = (min(EyesPoints, key=lambda item: item[1]))[1]
    # print(EyesPoints, '    ', minX, minY, maxX, maxY)
    eye[mask == 0] = 255
    # Croping Eye from the frame
    cropedEye = eye[minY:maxY, minX:maxX]
    height, width = cropedEye.shape
    div = int(width/3)

    ret, threshEye = cv.threshold(cropedEye, 100, 255, cv.THRESH_BINARY)

    rightPart = threshEye[0:height, 0:div]
    center = threshEye[0:height, div:div+div]
    leftPart = threshEye[0:height, div+div:div+div+div]
    rightBlack = np.sum(rightPart == 0)
    centerBlack = np.sum(center == 0)
    leftBlack = np.sum(leftPart == 0)
    EyePosition = [rightBlack, centerBlack, leftBlack]
    # print(rightBlack, "   ", centerBlack, "  ", leftBlack)
    # if rightBlack > centerBlack and rightBlack > leftBlack:
    #     EyePosition = 'Right'

    # elif leftBlack > rightBlack and leftBlack > centerBlack:
    #     EyePosition = 'Left'
    # elif centerBlack > rightBlack and centerBlack > leftBlack:
    #     # print("Center")
    #     EyePosition = 'Center'
    # else:
    #     EyePosition = 'Closed'

    return mask, threshEye, EyePosition


def Position(ValuesList):

    maxIndex = ValuesList.index(max(ValuesList))

    if maxIndex == 0:
        return "Right"
    elif maxIndex == 1:
        return "Center"
    elif maxIndex == 2:
        return "Left"
    else:
        return "Eye Closed"
