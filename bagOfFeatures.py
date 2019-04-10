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


#%% Global Vars

TRAIN_DATA = '/home/aries/ds_course/term2/compv/coursework/cropped/train/'
TEST_DATA  = '/home/aries/ds_course/term2/compv/coursework/cropped/test/'

f = "IMG_20190128_201734.jpg"
g = "IMG_20190128_201734.mp4"

NUM_CLASSES = 3
K = NUM_CLASSES * 10  # KMeans Classes

#%% Helper Functions

def getSIFTfeatures(gray_img):
    sift = cv2.xfeatures2d.SIFT_create() #What threshold?
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc

def getSURFfeatures(gray_img):  
    surf = cv2.xfeatures2d.SURF_create() #what threshold?
    surf.setExtended(True) # --> to expand to 128 dim.
    kp, desc = surf.detectAndCompute(gray_img, None)
    return kp, desc


def getBRIEFfeatures(gray_img):
    # Initiate FAST detector
    star = cv2.xfeatures2d.StarDetector_create()
    # Initiate BRIEF extractor
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    # find the keypoints with STAR
    kp = star.detect(gray_img, None)
    # compute the descriptors with BRIEF
    kp, desc = brief.compute(gray_img, kp)
    return kp, desc

def getORBfeatures(gray_img):
    # Initiate ORB detector
    orb = cv2.ORB_create()
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
        descriptors[i] = desc
        
    return descriptors



def generateHistograms(descriptors):
    
    des = list(descriptors.values())
    desc_matrix = reduce(lambda x,y: np.concatenate((x,y)), des)

    # Train KMeans
    initial_size = 3 * K
    batch_size = len(descriptors.keys()) * 3  # What's a good metric to determine this number? 
    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=batch_size, init_size=initial_size, verbose=0).fit(desc_matrix)
    #print("Finished K-Means")
    
    ## Generate Histogram 
    histograms = []
    for i, desc in descriptors.items(): 
        preds = kmeans.predict(desc)
        hist, bin_edges=np.histogram(preds, bins=range(0, K)) # Normalize by number of keypoints?? 
        histograms.append(hist)
    
    return histograms


#%% Final Feature Vectors

sift_train_feature_vec = [[], []]   
sift_test_feature_vec = [[], []]   

surf_train_feature_vec = [[], []]   
surf_test_feature_vec = [[], []]   

orb_train_feature_vec = [[], []]   
orb_test_feature_vec = [[], []]   

#%% Process each folder    

print("Training Data")
people = os.listdir(TRAIN_DATA)    
    
for person in people:
    
    print("Processing Sub Dir", person)
    
    #TRAIN DATA
    
    #read in cropped data
    crop_names = os.listdir(os.path.join(TRAIN_DATA, person))
    crop_names = list(map(lambda x: os.path.join(TRAIN_DATA, person, x), crop_names)) 
    crops = [cv2.imread(x , cv2.IMREAD_GRAYSCALE) for x in crop_names ]
    
        
    # get SIFT Features
    sift_desc = extractFeatures(crops, getSIFTfeatures)
    sift_histograms = generateHistograms(sift_desc)
    sift_train_feature_vec[0].extend(sift_histograms)
    sift_train_feature_vec[1].extend([person] * len(sift_histograms))
    
    # get SURF Features
    surf_desc = extractFeatures(crops, getSURFfeatures)
    surf_histograms = generateHistograms(surf_desc)
    surf_train_feature_vec[0].extend(surf_histograms)
    surf_train_feature_vec[1].extend([person] * len(surf_histograms))
    
    
    # get ORB Features
    orb_desc = extractFeatures(crops, getORBfeatures)
    orb_histograms = generateHistograms(orb_desc)
    orb_train_feature_vec[0].extend(orb_histograms)
    orb_train_feature_vec[1].extend([person] * len(orb_histograms))
    
    
    
print("Test Data")
people = os.listdir(TEST_DATA) 

for person in people:
    #TEST DATA
    
    print("Processing Sub Dir", person)
    
    #read in cropped data
    crop_names = os.listdir(os.path.join(TEST_DATA, person))
    crop_names = list(map(lambda x: os.path.join(TEST_DATA, person, x), crop_names)) 
    crops = [cv2.imread(x , cv2.IMREAD_GRAYSCALE) for x in crop_names ]
    
        
    # get SIFT Features
    sift_desc = extractFeatures(crops, getSIFTfeatures)
    sift_histograms = generateHistograms(sift_desc)
    sift_test_feature_vec[0].extend(sift_histograms)
    sift_test_feature_vec[1].extend([person] * len(sift_histograms))
    
    # get SURF Features
    surf_desc = extractFeatures(crops, getSURFfeatures)
    surf_histograms = generateHistograms(surf_desc)
    surf_test_feature_vec[0].extend(surf_histograms)
    surf_test_feature_vec[1].extend([person] * len(surf_histograms))
    
    # get ORB Features
    orb_desc = extractFeatures(crops, getORBfeatures)
    orb_histograms = generateHistograms(orb_desc)
    orb_test_feature_vec[0].extend(orb_histograms)
    orb_test_feature_vec[1].extend([person] * len(orb_histograms))
    

    
##%% Train SVM
# 
#print("Training SVM")
#svmModel = svm.SVC(gamma = 'auto')
#svmModel.fit(sift_train_feature_vec[0], sift_train_feature_vec[1])
#
#
##%% Train MLP
#
#print("Training MLP")
#mlpModel = MLPClassifier(verbose=True, max_iter=6000)
#mlpModel.fit(sift_train_feature_vec[0], sift_train_feature_vec[1])
#
#
##%% Predict on Test (SVM)
#svmPreds = svmModel.predict(sift_test_feature_vec[0])
#svmCM = confusion_matrix(sift_test_feature_vec[1], svmPreds)
#
#
##%% Predict on Test (MLP)
#
#mlpPreds = mlpModel.predict(sift_test_feature_vec[0])
#mlpCM = confusion_matrix(sift_test_feature_vec[1], mlpPreds)



     
# =============================================================================
# sources consulted : 
#
# https://www.superdatascience.com/blogs/opencv-face-detection
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html
# https://docs.opencv.org/3.4/df/dd2/tutorial_py_surf_intro.html
# https://ianlondon.github.io/blog/how-to-sift-opencv/
# https://www.kaggle.com/pierre54/bag-of-words-model-with-sift-descriptors/notebook
# https://stackoverflow.com/questions/51168896/bag-of-visual-words-implementation-in-python-is-giving-terrible-accuracy
# =============================================================================


