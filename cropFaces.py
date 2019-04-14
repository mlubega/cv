#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 18:58:21 2019

@author: aries
"""

import imutils
from imutils import paths
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys
import os
import random 
from sklearn.model_selection import train_test_split
import logging


logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('cropFaces.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)

RAND_COUNT = 30
VID_SAMPLE = 10
NUM_CLASSES = 0
IMG_EXTS = ['.jpg', '.jpeg']
VID_EXTS = ['.mp4', '.mov']

face_detector = cv2.CascadeClassifier('/home/aries/opencv-4.0.1/data/haarcascades/haarcascade_frontalface_default.xml')
src_path = '/home/aries/ds_course/term2/compv/coursework/test/'
dst_path = '/home/aries/ds_course/term2/compv/coursework/tcrop/'


def detectFaceAndCrop(img):
    
    cropped_faces = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    if (len(faces) > 1):
        logger.debug("%i bboxes found in image", len(faces))
    for (x,y,w,h) in faces:     
        #(x,y,w,h) = increaseBBOX(face_box, 10) 
        crop = img[y:y+h, x:x+w]
        cropped_faces.append(crop)
        
    return cropped_faces

def writeCrops(crop_list, dest):
    if not os.path.isdir(dest):
        os.makedirs(dest)
            
    for i in range(0, len(crop_list)):
        filename = str(i) + ".jpg"
        cv2.imwrite(os.path.join(dest, filename), crop_list[i])      
    return
        

def generateCleanData():
        
    people = os.listdir(src_path) 
    NUM_CLASSES = len(people)
    
    for person in people:
        
        logger.debug("Processing Person %s ----------",  person)
        full_img_dir = os.path.join(src_path, person)
        jpgs = list(filter(lambda x: x.endswith(tuple(IMG_EXTS)), os.listdir(full_img_dir)))
        vids = list(filter(lambda x: x.endswith(tuple(VID_EXTS)), os.listdir(full_img_dir)))
    
        ## holds cropped images from both img & vids
        crop_shots = []
    
            
        #%% read  crop image file
        for pic in jpgs:
            img = cv2.imread(os.path.join(src_path, person, pic))
            crop_shots += detectFaceAndCrop(img)     
        logger.debug("%i cropped images from from images", len(crop_shots))
        
        #%% read and crop video files
        for vid in vids:
    
            cap = cv2.VideoCapture(os.path.join(src_path, person, vid))
             
            if not cap.isOpened:
                sys.exit('Cap Not Open')
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_size = min(RAND_COUNT, total_frames)
            
            frames_to_sample = random.sample(range(1, total_frames), sample_size)
            #print("Sampling", sample_size,"of", total_frames, "total frames")
            
            
            crops_taken = 0
            for fr in frames_to_sample:
                cap.set(1, fr)
                hasframe, frame = cap.read()
                
                if not hasframe: 
                    #print("Frame Not Found, loop  until a frame is found")
                    continue;
                    
                if vid.endswith('.mov'):
                    frame = imutils.rotate_bound(frame, 90)
                    
                shots = detectFaceAndCrop(frame)
                if not shots:
                    #print("Crop Not Taken, loop  until a crop is found")
                    continue;
                crop_shots += shots
                crops_taken += len(shots)
                if crops_taken >= VID_SAMPLE:
                    break;
                
            logger.debug("%i cropped images found from video %s", crops_taken, vid)
            cap.release
        
        logger.debug("%i found for person %s \n\n",  len(crop_shots),  person)
            
            
            
        #%% make train/test 
        train, test = train_test_split(crop_shots, train_size=0.8, random_state=45, shuffle=True)
        
        #%% Write cropped images
        writeCrops(train, os.path.join(dst_path, "train", person))
        writeCrops(test, os.path.join(dst_path, "test", person))
      


def main():
    generateCleanData()
    
main()
    