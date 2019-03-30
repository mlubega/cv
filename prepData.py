# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%% load libraries

from imutils import paths
import imutils
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy
import sys
import os

#%% Global Vars

face_detector = cv2.CascadeClassifier('/home/aries/opencv-4.0.1/data/haarcascades/haarcascade_frontalface_default.xml')
src_images = '/homes/aries/ds_course/term2/compv/coursework/test/'
dst_images = '/homes/aries/ds_course/term2/compv/coursework/cropped/'


#%% Iterate Over Dir

image_dirs = os.listdir(src_images)

f = "IMG_20190128_201734.jpg"
g = "IMG_20190128_201734.mp4"



#%% Helper Functions

def convertToRGB(img): 
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def increaseBBOX( bbox, pixels):
    x, y, w, h = bbox
    x = x - pixels
    w = w + (2 * pixels)
    y = y - pixels
    h = h + (2 * pixels)
    return (x,y,w,h)


def detectFace(img):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for face_box in faces:
        
        (x,y,w,h) = increaseBBOX(face_box, 10)

 
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)   
        cv2.imshow('img', img)
        
        if cv2.waitKey(0) & 0xFF == 27:
            break
    
    
    cv2.destroyAllWindows()
    return 0



#%% read and crop video files


cap = cv2.VideoCapture(g)
 
if not cap.isOpened:
    sys.exit('Cap Not Open')
     
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("# Frames: ", num_frames)
       
       
while(cap.isOpened):
    hasframe, frame = cap.read()
     
    if not hasframe:
        print("No Frame ")
        break;
    
    detectFace(frame)


cap.release()


#%% read  crop image files

img = cv2.imread(f)
cv2.namedWindow("img", cv2.WINDOW_NORMAL)  
cv2.resizeWindow('img', 600,600)
detectFace(img)

#%% Matplotlib 

#image = mpimg.imread(f)
#plt.imshow(image)
#plt.show()



