import argparse as ap
import cv2
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *

def classify(im):
    if im == None:
        print "No such file {}\nCheck if the file exists".format(image_path)
        return -1

    # Load the classifier, class names, scaler, number of clusters and vocabulary 
    clf, classes_names, stdSlr, k, voc = joblib.load("bow.pkl")

    sift = cv2.xfeatures2d.SIFT_create()

    kpts, des = sift.detectAndCompute(im, None)

    test_features = np.zeros((1, k), "float32")
    
    # words, distance = vq(des_list[0][1],voc)
    words, distance = vq(des,voc)
    for w in words:
        test_features[0][w] += 1

    # Perform tf-idf vectorization
    nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
    idf = np.array(np.log((1.0*1+1) / (1.0*nbr_occurences + 1)), 'float32')

    # Scale the features
    test_features = stdSlr.transform(test_features)

    # Perform the predictions
    predictions =  [classes_names[i] for i in clf.predict(test_features)]
    print predictions
    return predictions
