# Computer Vision Project Spring 2019

This project implement facials recognition models using both traditional and deep learning methods. It also has a naive algorithm for Optical Character Recognition (OCR). 


1 - bagOfFeatures.py

This is a basic implementation of the Bag of Visual Features  (BoVW) algorithm. It extracts SIFT, SURF, and ORB features and trains MLP and SVM classifiers. 

2 - TrainFacialRecognition.ipynb

This is the same implemenation of the Bag of Visual Features algorithm adapted to run in a Colab Notebook. Also includes an implementation of a VG166-based CNN for facial recognition using Keras. Colaboratory gives access to NVIDIA GPU which was necessary to speed up training when using the full data set. 

3 - orc.py
This is a naive algorithm for Optical Character Recognition (OCR). It uses morphological transformations (thresholding, erosion, dilation) and MSER regions for text detection, and Pytesseract for text recognition. 

4 - DetectNum.ipynb
This is the same algorithm from orc.py adapted for a Colaboratory Notebook. 

5 - cropFaces.py
This file processes image and video raw data to extract faces, fix rotation, and generate train/test splits. 

