import cv2
import numpy as np

'''
Capture images for training
'''

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    passret, img = cap.read()

    # assistant rectangle
    cv2.rectangle(img, (100,100), (500,500), (0,255,0))

    # valid area
    crop_img = img[100:500, 100:500]

    drawing = np.zeros(crop_img.shape, np.uint8)
    all_img = np.hstack((drawing, crop_img))

    cv2.imshow('image', img)

    edges = cv2.Canny(crop_img,100,200)

    cv2.imshow('Contours', edges)

    cmd = cv2.waitKey(10)
    # exit if press ESC
    if cmd == 27:
        break
    # capture frame by 'c'
    elif cmd == 99:
    	print 'capture image!'
    	cv2.imwrite	( "./edge_model/model.jpg", crop_img);
    	# cv2.imwrite	( "./edge_model/model.jpg", edges);