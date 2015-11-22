import argparse as ap
import cv2
import imutils 
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *

# Load the classifier, class names, scaler, number of clusters and vocabulary 
clf, classes_names, stdSlr, k, voc = joblib.load("bow.pkl")

sift = cv2.xfeatures2d.SIFT_create()

im = cv2.imread('t.png')

kpts, des = sift.detectAndCompute(im, None)
# des_list.append((image_path, des))

test_features = np.zeros((1, k), "float32")

words, distance = vq(des,voc)
for w in words:
    test_features[0][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scale the features
test_features = stdSlr.transform(test_features)

# Perform the predictions
predictions =  [classes_names[i] for i in clf.predict(test_features)]
print predictions