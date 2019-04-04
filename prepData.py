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
import random 

#%% Global Vars

face_detector = cv2.CascadeClassifier('/home/aries/opencv-4.0.1/data/haarcascades/haarcascade_frontalface_default.xml')
src_path = '/home/aries/ds_course/term2/compv/coursework/test/'
dst_path = '/home/aries/ds_course/term2/compv/coursework/cropped/'

f = "IMG_20190128_201734.jpg"
g = "IMG_20190128_201734.mp4"

VID_SAMPLE = 3

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


def detectFaceAndCrop(img):
    
    cropped_faces = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    print('Found', len(faces), 'faces to crop')
    for face_box in faces:     
        (x,y,w,h) = increaseBBOX(face_box, 10) 
        crop = img[y:y+h, x:x+w]
        cropped_faces.append(crop)

        #cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)   
        #cv2.imshow('img', img)
        #if cv2.waitKey(0) & 0xFF == 27:
        #    break
    #cv2.destroyAllWindows()
    return cropped_faces

def getSIFTfeatures(gray_img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc

#%% Create Sub Directories for cropped photos

#people = os.listdir(src_path)

#for person in people:
#    dst_dir = os.path.join(dst_path, person)
#    if not os.path.exists(dst_dir):
#        os.mkdir(dst_dir)
#    
#%% Process each folder    
    
for person in people:
    
    person_dir = os.path.join(src_path, person)
    cropped_dir = os.path.join(dst_path, person)
    jpgs = list(filter(lambda x: x.endswith('.jpg'), os.listdir(person_dir)))
    vids = list(filter(lambda x: x.endswith('.mp4'), os.listdir(person_dir)))

    ## Save cropped images
    crop_shots = []

    
        
    #%% read  crop image file
    for pic in jpgs:
        img = cv2.imread(os.path.join(src_path, person, pic))
        crop_shots += detectFaceAndCrop(img)
            
        #cv2.namedWindow("img", cv2.WINDOW_NORMAL)  
        #cv2.resizeWindow('img', 600,600)
        
    #%% read and crop video files
    for vid in vids:

        cap = cv2.VideoCapture(os.path.join(src_path,person, vid))
         
        if not cap.isOpened:
            sys.exit('Cap Not Open')
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("# Frames: ", total_frames)

        frames_to_sample = random.sample(range(1, total_frames), VID_SAMPLE)
        
        for fr in frames_to_sample:
            cap.set(1, fr)
            hasframe, frame = cap.read()
            if not hasframe: 
                print("No Frame")
                continue;
            crop_shots += detectFaceAndCrop(frame)
            
        cap.release
                   
#        while(cap.isOpened):
#            
#           
#            hasframe, frame = cap.read()
#             
#            if not hasframe:
#                print("No Frame ")
#                break;
#            
#            detectFace(frame)
#        
#        
#        cap.release()



#%% Check SIFT Features
#        
#img = cv2.imread(f)
##gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#crop_list = []
#faces = face_detector.detectMultiScale(img, 1.3, 5)
#for face_box in faces:
#    (x,y,w,h) = increaseBBOX(face_box, 10)
#    facecrop = img[y:y+h, x:x+w]
#    crop_list.append(facecrop)
#    
##cv2.imshow('i', crop_list[0]) 
##cv2.waitKey(0) 
##cv2.destroyAllWindows()
#
#sift = cv2.xfeatures2d.SIFT_create()
#gray = cv2.cvtColor(crop_list[0], cv2.COLOR_BGR2GRAY)
#kp, desc = sift.detectAndCompute(gray, None)
#kp_img = cv2.drawKeypoints(gray, kp, crop_list[0].copy())
#cv2.imshow('i', kp_img) 
#cv2.waitKey(0) 
#cv2.destroyAllWindows()
#
#
##%% Matplotlib 
#        
#        
#        
#        
#
#image = mpimg.imread( f)
#plt.imshow(image)
##plt.show()'
#
##%% 
#
#
#
#
#def show_sift_features(gray_img, color_img, kp):
#    cv2.namedWindow("img", cv2.WINDOW_NORMAL)  
#    cv2.resizeWindow('img', 600,600)
#    kp_img = cv2.drawKeypoints(gray_img, kp, color_img.copy())
#    cv2.imshow('img', kp_img)
#    cv2.waitKey(0)
#        
#    return
#
## generate SIFT keypoints and descriptors
#octo_front_kp, octo_front_desc = gen_sift_features(octo_front_gray)
#octo_offset_kp, octo_offset_desc = gen_sift_features(octo_offset_gray)
#
#print 'Here are what our SIFT features look like for the front-view octopus image:'
#show_sift_features(octo_front_gray, octo_front, octo_front_kp);
#        
#        
###sources used : https://ianlondon.github.io/blog/how-to-sift-opencv/


