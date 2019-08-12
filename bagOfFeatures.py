# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%% load libraries

import cv2
import numpy as np
import os
from functools import reduce
from sklearn.cluster import MiniBatchKMeans
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score

#%% Global Vars

TRAIN_DATA = '/home/aries/ds_course/term2/compv/coursework/test_crops/train/'
TEST_DATA  = '/home/aries/ds_course/term2/compv/coursework/test_crops/test/'



NUM_CLASSES = len(os.listdir(TRAIN_DATA))
K = NUM_CLASSES * 5  # KMeans Classes
sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
# Initiate FAST detector
star = cv2.xfeatures2d.StarDetector_create()
# Initiate BRIEF extractor
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
# Initiate ORB detector
orb = cv2.ORB_create()

#%% Helper Functions

def getSIFTfeatures(gray_img):
     #What threshold?
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc

def getSURFfeatures(gray_img):  
     #what threshold?
    surf.setExtended(True) # --> to expand to 128 dim.
    kp, desc = surf.detectAndCompute(gray_img, None)
    return kp, desc


def getBRIEFfeatures(gray_img):

    # find the keypoints with STAR
    kp = star.detect(gray_img, None)
    # compute the descriptors with BRIEF
    kp, desc = brief.compute(gray_img, kp)
    return kp, desc

def getORBfeatures(gray_img):

    # find the keypoints with ORB
    kp = orb.detect(gray_img,None)
    # compute the descriptors with ORB
    kp, desc = orb.compute(gray_img, kp)
    return kp, desc

def extractFeatures(crop_list, featureFunc):
    descriptors = {}
    
    for i in range(0, len(crop_list)):
       # gray = cv2.cvtColor(crop_list[i], cv2.COLOR_BGR2GRAY)
        kp, desc = featureFunc(crop_list[i])
        if type(desc) != type(None):
            descriptors[i] = desc
        
    return descriptors


def groupAllFeatures(desc_hash):
    all_desc = []
    for person, crops in desc_hash.items():
        for idx, descs in crops.items():
            all_desc.extend(descs)
            
    return all_desc
        

def trainKMeans(descriptors):
     
    # Train KMeans
    initial_size = 3 * K
    batch_size = int(len(descriptors) / 3)  # What's a good metric to determine this number? 
    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=batch_size, init_size=initial_size, verbose=0).fit(descriptors)
    
    return kmeans
    
def generateHistograms(descriptors, kmeans_centers):
    
    ## Generate Histogram 
    histograms = []
    for i, desc in descriptors.items(): 
        preds = kmeans_centers.predict(desc)
        hist, bin_edges=np.histogram(preds, bins=range(0, K)) # Normalize by number of keypoints?? 
        histograms.append(hist)
    
    return histograms


#%% Final Feature Vectors

sift_train_feature_vec = [[], []]   
sift_test_feature_vec = [[], []]  
sift_kmeans = None

surf_train_feature_vec = [[], []]   
surf_test_feature_vec = [[], []]  
surf_kmeans = None

orb_train_feature_vec = [[], []]   
orb_test_feature_vec = [[], []]   
orb_kmeans = None

#%% Process each folder    

print("Training Data")
people = os.listdir(TRAIN_DATA)    


allSIFT = {}
allSURF = {}
allORB = {}

#%% Extract Features

for person in people:
    
    print("Processing Sub Dir", person)
    
    
    
    #read in cropped data
    crop_names = os.listdir(os.path.join(TRAIN_DATA, person))
    crop_names = list(map(lambda x: os.path.join(TRAIN_DATA, person, x), crop_names)) 
    crops = [cv2.imread(x , cv2.IMREAD_GRAYSCALE) for x in crop_names ]
    
    # get SIFT Features
    sift_desc = extractFeatures(crops, getSIFTfeatures)
    allSIFT[person] = sift_desc
    print("Extracted SIFT")
    
    # get SURF Features
    surf_desc = extractFeatures(crops, getSURFfeatures)
    allSURF[person] = surf_desc
    print("Extracted SURF")
    
    # get ORB Features
    orb_desc = extractFeatures(crops, getORBfeatures)
    allORB[person] = orb_desc
    print("Extracted ORB")

#%% Group Features

sift_matrix = groupAllFeatures(allSIFT)
surf_matrix = groupAllFeatures(allSURF)
orb_matrix = groupAllFeatures(allORB)

#%% Train KMeans

print("Training SIFT KMeans")
sift_kmeans = trainKMeans(sift_matrix)

print("Training SURF KMeans")
surf_kmeans = trainKMeans(surf_matrix)

print("Training ORB KMeans")
orb_kmeans = trainKMeans(orb_matrix)
 
#%% Create Train Histograms    
    
for person in people:
    
    print("Creating Train Histograms for", person)
    
    sift_histograms = generateHistograms(allSIFT[person], sift_kmeans)
    sift_train_feature_vec[0].extend(sift_histograms)
    sift_train_feature_vec[1].extend([person] * len(sift_histograms))
    print("Extracted SIFT")
    
    # get SURF Features
    surf_histograms = generateHistograms(allSURF[person], surf_kmeans)
    surf_train_feature_vec[0].extend(surf_histograms)
    surf_train_feature_vec[1].extend([person] * len(surf_histograms))
    print("Extracted SURF")
    
    # get ORB Features
    orb_histograms = generateHistograms(allORB[person], orb_kmeans)
    orb_train_feature_vec[0].extend(orb_histograms)
    orb_train_feature_vec[1].extend([person] * len(orb_histograms))
    print("Extracted ORB")
    
    
    
print("Test Data")
people = os.listdir(TEST_DATA) 

for person in people:
    #TEST DATA
    
    print("Creating Test Histograms for", person)
    
    
    # get SIFT Features
    sift_histograms = generateHistograms(allSIFT[person], sift_kmeans)
    sift_test_feature_vec[0].extend(sift_histograms)
    sift_test_feature_vec[1].extend([person] * len(sift_histograms))
    print("Extracted SIFT")
    
    # get SURF Features
    surf_histograms = generateHistograms(allSURF[person], surf_kmeans)
    surf_test_feature_vec[0].extend(surf_histograms)
    surf_test_feature_vec[1].extend([person] * len(surf_histograms))
    print("Extracted SURF")
    
    # get ORB Features
    orb_histograms = generateHistograms(allORB[person], orb_kmeans)
    orb_test_feature_vec[0].extend(orb_histograms)
    orb_test_feature_vec[1].extend([person] * len(orb_histograms))
    print("Extracted ORB")
    

    
##%% Train SVM
 
print("Training SVM")
svmModel = svm.SVC(gamma = 'auto')
svmModel.fit(sift_train_feature_vec[0], sift_train_feature_vec[1])


#%% Train MLP

print("Training MLP")
mlpModel = MLPClassifier(verbose=False, max_iter=6000)
mlpModel.fit(sift_train_feature_vec[0], sift_train_feature_vec[1])
#
#
##%% Predict on Test (SVM)
svmPreds = svmModel.predict(sift_test_feature_vec[0])
svmScore = accuracy_score(sift_test_feature_vec[1], svmPreds)
print("SVM Score:", svmScore)


#%% Predict on Test (MLP)

mlpPreds = mlpModel.predict(sift_test_feature_vec[0])
mlpScore = accuracy_score(sift_test_feature_vec[1], mlpPreds)
print("MLP Score:", mlpScore)


     


