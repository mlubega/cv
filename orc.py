#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:34:27 2019

@author: aries
"""

import cv2 
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import imutils
import sys

#img_path = "sophie_side.jpeg"
#img_path = "jack_side.jpeg"# --> not so good
#img_path = "jack.jpeg"#
#img_path = "sepher.jpg"#

#img_path = "./IMG_3546.jpeg"
#img_path = "./IMG_3323.jpg"   #Eijaz
#img_path = "./IMG_3542.jpeg"
#img_path = "./IMG_20190128_201734.jpg"
vid_path = "./IMG_20190128_201734.mp4"
#img_path = "./IMG_20190128_202452.jpg"  # Juan
#vid_path = './IMG_3542.mov'



def showImage(img):
    cv2.namedWindow("hunk", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('hunk', 600,600)
    cv2.imshow('hunk', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



cap = cv2.VideoCapture(vid_path)
             
if not cap.isOpened:
   sys.exit('Cap Not Open')
                
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frames:",total_frames)

pic_found = 0 

for i in range(total_frames):
    hasframe, frame = cap.read()
    
    if not hasframe:
        continue;
        
    if vid_path.endswith('.mov'):
        frame = imutils.rotate_bound(frame, 90)
        
    pic_found = frame
    break;

img = pic_found

#%% set mser size

#img =  cv2.imread(img_path)
img_height, img_width = img.shape[:2]
img_area = float(img_height) * img_width
max_num_area = int(img_area * 0.004)
kernel_edge = int(np.sqrt(max_num_area / 2))

print("Shape:", img.shape)
print("Area:", img_area)
print("Calculated MSER area:",max_num_area)
print("Calculated Kernel:",kernel_edge)



#%% Threshold
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
retval, graythresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#showImage(graythresh)
#plt.imshow(graythresh)
#plt.show()


#%% Opening
kernel = np.ones((kernel_edge,kernel_edge),np.uint8)
opening = cv2.morphologyEx(graythresh, cv2.MORPH_OPEN, kernel)
#plt.imshow(opening)
#plt.show()


#showImage(opening)
#%% MSER

rois = []

vis = img.copy()
mser = cv2.MSER_create(_delta=10, _max_area=max_num_area, _max_variation=0.5)
regions, _ = mser.detectRegions(opening)
for p in regions:
    xmax, ymax = np.amax(p, axis=0)
    xmin, ymin = np.amin(p, axis=0)
    
    
    xmax = min(xmax + 30, img_width )
    ymax = min(ymax + 30, img_height)
    xmin = max(xmin - 30, 1)
    ymin = max(ymin - 30, 1)


    rois.append(img[ymin:ymax, xmin:xmax])
    cv2.rectangle(vis, (xmin,ymax), (xmax,ymin), (0, 255, 0), 1)

  
showImage(vis)
#plt.imshow(vis)
#plt.show()

print(len(rois), "possible images")

#%% filtering by histogram
best_roi = None
best_pct = .88
for img in rois:
  plt.imshow(img)
  plt.show()
  #showImage(img)
  gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  hist,bins = np.histogram(gimg.ravel(),16)
  
  #print(hist)

  low = sum(hist[:4])
  high = sum(hist[12:])
  al = sum(hist)
  bimodal_pct = (low+high)/al
  print(bimodal_pct)
  
  plt.hist(img.ravel(),16)
  plt.show()
  
  if bimodal_pct > best_pct:
      best_roi = img
      best_pct = bimodal_pct



if best_roi is None:
    sys.exit("No num detected")

print("BEST ROI")
plt.imshow(best_roi)
plt.show()

#%% deskew
g_roi = cv2.cvtColor(best_roi, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(g_roi)

# threshold the image, setting all foreground pixels to
# 255 and all background pixels to 0
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

#cv2_imshow(thresh)

# grab the (x, y) coordinates of all pixel values that
# are greater than zero, then use these coordinates to
# compute a rotated bounding box that contains all
# coordinates
coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]
print(angle)
 
# the `cv2.minAreaRect` function returns values in the
# range [-90, 0); as the rectangle rotates clockwise the
# returned angle trends to 0 -- in this special case we
# need to add 90 degrees to the angle
if angle < -45:
	angle = -(90 + angle)
 
# # otherwise, just take the inverse of the angle to make
# # it positive
else:
	angle = -angle
  
print(angle)


# rotate the image to deskew it
image = best_roi.copy()
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(image, M, (w, h),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)



ret, thresh1 = cv2.threshold(rotated, 150,255,cv2.THRESH_BINARY)
plt.imshow(thresh1)
plt.show()

#%% pytesseract



config = ("-l eng --oem 1 --psm 6  outputbase digits")
#config = ("-l eng --oem 1 --psm 6 -c tessedit_char_whitelist=-:0123456789")
text = pytesseract.image_to_string(thresh1, config=config) 
print(text)



# https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
# https://pythonprogramming.net/morphological-transformation-python-opencv-tutorial/?completed=/blurring-smoothing-python-opencv-tutorial/

