import cv2 as cv 
import numpy as np 
import dlib
# variables

# colors 
YELLOW = (0,255,255)
CYAN = (255,255,0)
MAGENTA =(255,0,255)
GOLDEN = (32,218,165)
LIGHT_BLUE = (255,30,144)
PURPLE = (128,0,128)
CHOCOLATE = (30,105,210)
PINK = (147,20,255)
ORANGE = (0,69,255)
# objects 
detectFace= dlib.get_frontal_face_detector()
# landmarks detector  
predictor = dlib.shape_predictor("Predictor/shape_predictor_68_face_landmarks.dat")

# function
def midpoint(pts1, pts2):
    x, y = pts1
    x1, y1 = pts2
def faceDetector(image, Draw =True):
    pts1=(0,0)
    pts2=(0,0)
    IDs=0
    Gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = detectFace(Gray)
    face= None
    for face  in faces:
        # print(face)
        pts1 = (face.left(), face.top())
        pts2 = (face.right(), face.bottom())
        # landmarks = predictor(Gray, face)
        # x=landmarks.part(36).x
        # y=landmarks.part(36).y
        # print(type(landmarks))
        if Draw ==True:
            cv.rectangle(image, pts1, pts2, (0,255,255), 1)
    # print(data)
    return image,face
def facePoint(grayImage, image,face, Draw=False):
    landmarks =predictor(grayImage, face,)
    pointsList = []
    for n in range(0,68):
        point = (landmarks.part(n).x, landmarks.part(n).y)
        pointsList.append(point)
        if Draw ==True:
            cv.circle(image,point, 2, ORANGE, 2)
    # print(pointsList)
    # print(point)
    return image,pointsList

def blinkDetector(eyePoints):



    
