# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%% load libraries


import imutils
from imutils import paths
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys
import os
import random 
import operator
from functools import reduce
from sklearn.cluster import MiniBatchKMeans
from sklearn.neural_network import MLPClassifier
from sklearn import svm


#%% Global Vars

face_detector = cv2.CascadeClassifier('/home/aries/opencv-4.0.1/data/haarcascades/haarcascade_frontalface_default.xml')
src_path = '/home/aries/ds_course/term2/compv/coursework/test/'
dst_path = '/home/aries/ds_course/term2/compv/coursework/cropped/'

f = "IMG_20190128_201734.jpg"
g = "IMG_20190128_201734.mp4"

RAND_COUNT = 10
VID_SAMPLE = 3
NUM_CLASSES = 3
IMG_EXTS = ['.jpg', '.jpeg']
VID_EXTS = ['.mp4', '.mov']
K = NUM_CLASSES * 10

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
    #print('Found', len(faces), 'faces to crop')
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

#%% Final Feature Vectors

feature_vec = []
label_vec = []    


#%% Process each folder    

people = os.listdir(src_path)    


    
for person in people:
    
    print("Processing Sub Dir", person)
    person_dir = os.path.join(src_path, person)
    cropped_dir = os.path.join(dst_path, person)
    jpgs = list(filter(lambda x: x.endswith(tuple(IMG_EXTS)), os.listdir(person_dir)))
    vids = list(filter(lambda x: x.endswith(tuple(VID_EXTS)), os.listdir(person_dir)))

    ## Save cropped images
    crop_shots = []

        
    #%% read  crop image file
    for pic in jpgs:
        img = cv2.imread(os.path.join(src_path, person, pic))
        crop_shots += detectFaceAndCrop(img)
        
        
    print(len(crop_shots), "cropped images from from images")
    #%% read and crop video files
    for vid in vids:

        cap = cv2.VideoCapture(os.path.join(src_path,person, vid))
         
        if not cap.isOpened:
            sys.exit('Cap Not Open')
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_sample = random.sample(range(1, total_frames), RAND_COUNT)
        print("Sampling", frames_to_sample,"of", total_frames, "frames")
        
        
        crops_taken = 0
        for fr in frames_to_sample:
            cap.set(1, fr)
            hasframe, frame = cap.read()
            if not hasframe: 
                #print("Frame Not Found, loop  until a frame is found")
                continue;
            shots = detectFaceAndCrop(frame)
            if not shots:
                #print("Crop Not Taken, loop  until a crop is found")
                continue;
            crop_shots += shots
            crops_taken += len(shots)
            if crops_taken == VID_SAMPLE:
                break;
            
        print(crops_taken, "cropped images found from video")
        cap.release
        
        
    #%% get SIFT Features
        
    sift_descriptors = {}
    
    for i in range(1, len(crop_shots)):
        kp, desc = getSIFTfeatures(crop_shots[i])
        sift_descriptors[i] = desc

        
    #%% Train KMeans
    
    des = list(sift_descriptors.values())
    sift_desc_matrix = reduce(lambda x,y: np.concatenate((x,y)), des)

    batch_size = len(crop_shots) * 3  # What's a good metric to determine this number? 
    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=batch_size, verbose=0).fit(sift_desc_matrix)
    print("Finished K-Means")
    
    
    #%% Generate Histogram
    
    for i, desc in sift_descriptors.items(): 
        preds = kmeans.predict(desc)
        hist, bin_edges=np.histogram(preds, bins=range(1, K)) # Normalize by number of keypoints?? 
        feature_vec.append(hist)
        label_vec.append(person)
        
        
        
        
        
            
#%% Train SVM
 
print("Training SVM")
svmModel = svm.SVC()
svmModel.fit(feature_vec, label_vec)


#%% Train MLP

print("Training MLP")
mlp = MLPClassifier(verbose=True, max_iter=6000)
mlp.fit(feature_vec, label_vec)

# Testing the MOV files issues ---- 
# =============================================================================
# mov_file = "IMG_3545.mov"
# cap = cv2.VideoCapture(os.path.join(src_path,'3', mov_file))
# if not cap.isOpened:
#     sys.exit('Cap Not Open')
#         
# while(cap.isOpened):
#     hasframe, frame = cap.read()
#     if not hasframe: 
#         print("Frame Not Found")
#         break;
#     
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_detector.detectMultiScale(gray, 1.3, 5)
#     for (x,y,w,h) in faces:     
#         cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)   
#         cv2.imshow('img', frame)
#         cv2.waitKey(5)
# 
# 
# cv2.destroyAllWindows()
# =============================================================================
     
# =============================================================================
# sources consulted : 
#
# https://ianlondon.github.io/blog/how-to-sift-opencv/
# https://www.kaggle.com/pierre54/bag-of-words-model-with-sift-descriptors/notebook
# https://stackoverflow.com/questions/51168896/bag-of-visual-words-implementation-in-python-is-giving-terrible-accuracy
# =============================================================================


