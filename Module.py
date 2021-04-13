import cv2 as cv 
import numpy as np 
import dlib

# variables

# colors 

# objects 
detectFace= dlib.get_frontal_face_detector()

# function

def faceDetector(image, Draw =True):
    pts1=(0,0)
    pts2=(0,0)
    IDs=0
    Gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = detectFace(Gray)
    for IDs, face in enumerate(faces):
        pts1 = (face.left(), face.top())
        pts2 = (face.right(), face.bottom())
        if Draw ==True:
            cv.rectangle(image, pts1, pts2, (0,255,255), 1)
            if IDs ==1:
                cv.rectangle(image,pts1, pts2, (255,255,255), 3)
    
    return image,IDs, pts1, pts2
  

