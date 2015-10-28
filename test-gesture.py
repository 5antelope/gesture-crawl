from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
import math

from consine_simi import consine_similarity

edges_model = []
model_path='./edge_model'
for f in listdir(model_path):
    if isfile(join(model_path,f)) and join(model_path,f).endswith('.jpg'):
    	full_path = join(model_path,f)
        edges_model.append(cv2.imread(full_path, 0))
print 'load %d edge modes' % (len(edges_model))

def matchEdge(f):
    best_match = -1
    min = 2
    print len(edges_model)
    for m in range(len(edges_model)):
        if abs(consine_similarity(f, edges_model[m])) < min:
            min = abs(consine_similarity(f, edges_model[m]))
            best_match = m
    return best_match

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, img = cap.read()

    # assistant rectangle
    cv2.rectangle(img, (100,100), (500,500), (0,255,0))
    
    # valid area
    crop_img = img[100:500, 100:500]


    edges = cv2.Canny(crop_img,100,200)
    match_edge = matchEdge(edges)
    print 'best match with edge %d' % (match_edge)

    value = (35, 35)
    
    # blur the frame to reduce noise
    gaussian_blur = cv2.GaussianBlur(\
        cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY), \
        value, \
        0)
    
    # THRESH_BINARY_INV: > threshold = 0; otherwise = 255
    # THRESH_OTSU: opt threshold
    _, silhouette = cv2.threshold(gaussian_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # silhouette of hand
    # cv2.imshow('Silhouette', silhouette)

    _, contours, hierarchy = cv2.findContours(silhouette, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    max_area = -1
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if(area>max_area):
            max_area=area
            ci=i

    point_set = contours[ci]

    x,y,w,h = cv2.boundingRect(point_set)

    hull = cv2.convexHull(point_set)
    
    drawing = np.zeros(crop_img.shape, np.uint8)

    # cv2.drawContours(drawing, [point_set], 0, (0,255,0), 0)
    # cv2.drawContours(drawing, [hull], 0, (0,0,255), 0)
    
    hull = cv2.convexHull(point_set, returnPoints = False)
    convexity_defects = cv2.convexityDefects(point_set, hull)

    defects_cnt = 0

    # cv2.drawContours(silhouette, contours, -1, (0,255,0), 3)
    
    for i in range(convexity_defects.shape[0]):
        s,e,f,d = convexity_defects[i,0]
        start = tuple(point_set[s][0])
        end = tuple(point_set[e][0])
        far = tuple(point_set[f][0])
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
        if angle <= 90:
            defects_cnt += 1
            cv2.circle(crop_img,far,1,[0,0,255],-1)
        
        cv2.line(crop_img, start, end, [0,255,0], 2)
        cv2.circle(crop_img, far, 5, [0,0,255], -1)

    if defects_cnt == 1:
        cv2.putText(img,"ONE", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif defects_cnt == 2:
        cv2.putText(img, "TWO", (5,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    elif defects_cnt == 3:
        cv2.putText(img,"THREE", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif defects_cnt == 4:
        cv2.putText(img,"FOUR", (5,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif defects_cnt == 5:	
        cv2.putText(img,"FIVE", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    else:
        cv2.putText(img,"Hello World", (50,50),\
                    cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    
    cv2.imshow('Gesture', img)
    # cv2.imshow('Edges', edges)
    all_img = np.hstack((drawing, crop_img))
    cv2.imshow('Contours', all_img)
    
    # exit if press ESC
    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()
