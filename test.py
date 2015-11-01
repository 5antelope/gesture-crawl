from os import listdir
from os.path import isfile, join

import cv2
import numpy

'''
how to read from directory
'''
model_path='./edge_model'
print listdir(model_path)
files = []
for f in listdir(model_path):
	print join(model_path,f)
 	if isfile(join(model_path,f)) and join(model_path,f).endswith('.jpg'):
 		files.append(f)
print len(files)

'''
try to match shape by matchShapes()
	* after canny or threshold process, cannot recognize two gestures
'''
img1 = cv2.imread('./edge_model/1.jpg',0)
img2 = cv2.imread('./edge_model/2.jpg',0)

ret, thresh = cv2.threshold(img1, 127, 255,0)
ret, thresh2 = cv2.threshold(img2, 127, 255,0)
_,contours,hierarchy = cv2.findContours(thresh,2,1)
cnt1 = contours[0]
_,contours,hierarchy = cv2.findContours(thresh2,2,1)
cnt2 = contours[0]

ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
print ret
while (True):
	cv2.imshow('1', thresh)
	cv2.imshow('2', thresh2)