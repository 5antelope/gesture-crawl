from os import listdir
from os.path import isfile, join

import cv2
import numpy

model_path='./edge_model'
print listdir(model_path)
files = []
for f in listdir(model_path):
	print join(model_path,f)
 	if isfile(join(model_path,f)) and join(model_path,f).endswith('.jpg'):
 		files.append(f)
print len(files)