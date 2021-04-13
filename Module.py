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
predictor = dlib.shape_predictor('Predictor/shape_predictor_68_face_landmarks.dat')

# function

def faceDetector(image, Draw =True):
    pts1=(0,0)
    pts2=(0,0)
    IDs=0
    Gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = detectFace(Gray)
    face= []
    for IDs, face in enumerate(faces):
        print(face)
        pts1 = (face.left(), face.top())
        pts2 = (face.right(), face.bottom())
        if Draw ==True:
            cv.rectangle(image, pts1, pts2, (0,255,255), 1)
            if IDs ==1:
                cv.rectangle(image,pts1, pts2, (255,255,255), 3)
    # print(data)
    return image,face

def BlinkingDetection(GrayImage,face ):
    landmarks = predictor(GrayImage, face)
    print('working')
    # firstPoint = (landmarks.part[42].x, landmarks.part[42].y)
    # print(firstPoint)



    
