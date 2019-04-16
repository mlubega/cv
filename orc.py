#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:34:27 2019

@author: aries
"""

import cv2 
import numpy as np
import pytesseract
import imutils



img_path = "./IMG_20190128_201734.jpg"
vid_path = "./IMG_20190128_201734.mp4"
east_dect = "./frozen_east_text_detection.pb"



img =  cv2.imread(img_path)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow("hunk", cv2.WINDOW_NORMAL)
cv2.resizeWindow('hunk', 600,600)



#hue ,saturation ,value = cv2.split(hsv)
#two = np.hstack((value, gray))
#retval, graythresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#retval, valuethresh = cv2.threshold(value, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#thresh =np.hstack((graythresh, valuethresh))
#th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 301, 10)

lower = np.array([200,200,200])
upper = np.array([255,255,255])

mask = cv2.inRange(img, lower, upper)
res = cv2.bitwise_and(img,img, mask= mask)


kernel = np.ones((100,100),np.uint8)
opening = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)

two = np.hstack((res, opening))
cv2.imshow('hunk',two)

cv2.waitKey(0)
cv2.destroyAllWindows()


# https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
# https://pythonprogramming.net/morphological-transformation-python-opencv-tutorial/?completed=/blurring-smoothing-python-opencv-tutorial/

