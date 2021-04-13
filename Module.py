import cv2 as cv 
import numpy as np 
import dlib

# variables

# colors 

# objects 
detectFace= dlib.get_frontal_face_detector()

# function

def faceDetector(image, Draw =True):
    Gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = detectFace(Gray)
    for IDs, face in enumerate(faces):
        X, Y = face.left(), face.top()
        X1, Y1 = face.right(), face.bottom()
        if Draw ==True:
            cv.rectangle(image, (X, Y), (X1, Y1), (0,255,255), 1)
    return X, Y, X1, Y1
  

