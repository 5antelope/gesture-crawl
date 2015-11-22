from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
import math

from consine_simi import consine_similarity
from classify import classify
from MotionDetector import directionCalculate

cap = cv2.VideoCapture(0)

def setbg():
    retval, im = cap.read()
    avg1 = np.float32(im)
    print "setting background"
    for i in range(100): 
         retval, im = cap.read()
         cv2.accumulateWeighted(im,avg1, 0.1)
         res1 = cv2.convertScaleAbs(avg1)
         cv2.waitKey(10)
    cv2.imshow("Background",res1)
    return res1

# function to subtract background and frames
def extract(imgbg,imgfg):
    # split the images into RGB channels 
    b1,g1,r1 = cv2.split(imgbg)
    b2,g2,r2 = cv2.split(imgfg)

    # find absolute difference between respective channels 
    bb = cv2.absdiff(b1,b2)
    gg = cv2.absdiff(g1,g2)
    rr = cv2.absdiff(r1,r2)

    # threshold each channel
    ret1, b = cv2.threshold(bb,50,255,cv2.THRESH_BINARY)
    ret2, g = cv2.threshold(gg,50,255,cv2.THRESH_BINARY) 
    ret3, r = cv2.threshold(rr,50,255,cv2.THRESH_BINARY)

    # merge and blur the image
    rgb = cv2.merge((r,g,b))
    cv2.medianBlur(rgb,3)
    return rgb

# main entry
imgbg = setbg()
first_frame = cv2.cvtColor(imgbg, cv2.COLOR_BGR2GRAY)

while(cap.isOpened()):
    ret, img = cap.read()
    
    diff = np.linalg.norm(cv2.absdiff(imgbg, img))
    if diff < 8000:
        continue

    # removed background
    crop_img = extract(imgbg,img)
    gray= cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    _, thresh1 = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas) 

    #extract biggest contour and topmost point of that
    cnt=contours[max_index]

    # area: (x,y), (x+w, y+h)
    x,y,w,h = cv2.boundingRect(cnt)
    
    drawing = np.zeros(crop_img.shape, np.uint8)
    cv2.drawContours(drawing, contours, max_index, (0,255,0), 2)
    cv2.rectangle(drawing, (x,y), (x+w, y+h), (0,255,0))

    cv2.imshow("tracking", drawing[y:y+h,x:x+w])

    hand_fig = img[y:y+h,x:x+w]
    
    # cv2.imshow('hand', hand_fig)
    prediction = classify(drawing[y:y+h,x:x+w])
    
    feature_params = dict( maxCorners = 100,
             qualityLevel = 0.3,
             minDistance = 7,
             blockSize = 7 )

    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if prediction[0] == 'one':
        cv2.putText(img,"MODEL ONE", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 5)
        direction = directionCalculate(first_frame, frame, np.array(cnt, dtype=np.float32))
        print 'one %s' % (direction)
    elif prediction[0] == 'two':
        cv2.putText(img,"MODEL TWO", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 5)
        direction = directionCalculate(first_frame, frame, np.array(cnt, dtype=np.float32))
        print 'two %s' % (direction)

    cv2.imshow('main', img)
    
    # exit if press ESC
    if cv2.waitKey(10) == 27:
        break

    first_frame = frame

cap.release()
cv2.destroyAllWindows()
