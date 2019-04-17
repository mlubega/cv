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
from functools import reduce
import shutil


logger = logging.getLogger('PrepData')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('prepdata.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
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
src_path = '/home/aries/ds_course/term2/compv/coursework/imgs/'
dst_path = '/home/aries/ds_course/term2/compv/coursework/crops/'


def detectFaceAndCrop(img):
    
    cropped_faces = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5,  minSize=(80, 80))
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
            #sample_size = min(RAND_COUNT, total_frames) 
            #frames_to_sample = random.sample(range(1, total_frames), sample_size)
            #print("Sampling", sample_size,"of", total_frames, "total frames")
            
            
            crops_taken = 0
            #for fr in frames_to_sample:
            for i in range(total_frames):
                #cap.set(1, fr)
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
#                if crops_taken >= VID_SAMPLE:
#                    break;
                
            logger.debug("%i cropped images found from video %s", crops_taken, vid)
            cap.release
        
        logger.debug("%i found for person %s \n\n",  len(crop_shots),  person)
            
            
        writeCrops(crop_shots, os.path.join(dst_path,person))
        
        
def balanceClasses(imgs_path):
    
    people = os.listdir(imgs_path) 
    
    full_paths = list(map(lambda x: os.path.join(imgs_path,  x), people))
    file_counts = list(map(lambda x: len(os.listdir(x)), full_paths))
    smallest_count = min(file_counts)
    
    for folder in full_paths:
        
        imgs = os.listdir(folder)
        in_count = len(imgs)
        random.seed(4)
        random.shuffle(imgs)
        to_delete = imgs[smallest_count:]
        for img in to_delete:
            os.remove(os.path.join(folder, img))
        fin_count = len(os.listdir(folder))
        logger.debug("Balanced %s from %i to %i imgs", os.path.basename(folder), in_count, fin_count)
        
      
def createDir(path):
   if not os.path.isdir(path):
         os.makedirs(path)
         
def moveFiles(file_list, orig_path, dest_path):
          
    createDir(os.path.join(dest_path))

    for f in file_list: 
        # Move a file from the directory d1 to d2
        src = os.path.join(orig_path, f)
        dst = os.path.join(dest_path, f)
        shutil.move(src, dst)  
        
def createTrainTest(imgs_path):
    
    people = os.listdir(imgs_path) 
    
    if ["train", "test"] in people:
        sys.exit("train and test directories already exist")
        
    for person in people:
        logger.debug("Splitting Train/Test %s", person)
        img_names = os.listdir(os.path.join(imgs_path, person))
        train, test = train_test_split(img_names, train_size=0.7, random_state=45, shuffle=True)
        
 
        moveFiles(train, os.path.join(imgs_path, person), os.path.join(imgs_path, "train", person))
        moveFiles(test, os.path.join(imgs_path, person), os.path.join(imgs_path, "test", person))
            
          
def cleanUpEmptyFiles(imgs_path):
    
    people = os.listdir(imgs_path)
    full_paths = list(map(lambda x: os.path.join(imgs_path, x), people))
    for folder in full_paths:
        if (len(os.listdir(folder)) == 0):
            os.rmdir(folder)
    
    logger.debug("Cleaned Up Empty Folder")

def main():
    generateCleanData()
    
    ## Must manually remove false positives before running the rest
    
    #balanceClasses(dst_path)
    #createTrainTest(dst_path)
    #cleanUpEmptyFiles(dst_path)
    logger.handlers = []
    
main()
    