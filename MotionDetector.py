import numpy as np
import cv2
import time
from math import atan2, degrees, pi, hypot

def directionCalculate(first_frame, frame, feature_points):

    # Calculate the zone of the movement directions
    def calcZone(x1, y1, x2, y2):
        zone = [-1, 0]
        dx = x2 - x1
        # dy = y2 - y1
        # lower is bigger, thus switch the values.
        dy = y1 - y2
        # distance
        zone[1] = hypot(dx, dy)
        rads = atan2(dx, dy)
        rads %= 2*pi
        degs = degrees(rads)
        if degs >= 350 or degs <= 10:
            zone[0] = 0
        elif degs > 10 and degs < 80:
            zone[0] = 1
        elif degs >=80 and degs <= 100:
            zone[0] = 2
        elif degs >100 and degs < 170:
            zone[0] = 3
        elif degs >= 170 and degs <= 190:
            zone[0] = 4
        elif degs > 190 and degs < 260:
            zone[0] = 5
        elif degs >= 260 and degs <= 280:
            zone[0] = 6
        elif degs > 280 and degs < 350:
            zone[0] = 7
        return zone

    def zoneStringRepresent(z):
        zs = ""
        if z == 0:
            zs = "up"
        elif z == 1:
            zs = "up right"
        elif z == 2:
            zs = "right"
        elif z == 3:
            zs = "down right"
        elif z == 4:
            zs = "down"
        elif z == 5:
            zs = "down left"
        elif z == 6:
            zs = "left"
        else:
            zs = "up left"
        return zs


    # Retrieve the carmera
    # cap = cv2.VideoCapture(0)

    # Params for ShiTomasi corner detection
    # feature_params = dict( maxCorners = 100,
    #              qualityLevel = 0.3,
    #              minDistance = 7,
    #              blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
              maxLevel = 2,
              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    # ret, first_frame = cap.read()
    # first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    first_gray = first_frame
    #p0 = cv2.goodFeaturesToTrack(first_gray, mask = None, **feature_params)
    p0 = feature_points
    # Create a mask image for drawing purposes
    # mask = np.zeros_like(first_frame)

    # start_timer = time.time()
    directs = [0] * 8;

    # while(True):
        
    
    # ret, frame = cap.read()
    # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = frame
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(first_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        # Calculate the zone
        zone = calcZone(c, d, a, b)
        if zone[0] == -1:
            print 'ERROR CALCULATING THE ZONE'
        else:
            directs[zone[0]] += zone[1]
        # mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        # frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    # img = cv2.add(frame,mask)
    # now = time.time()
    # if (now-start_timer) > 2:
    for index in range(8):
        if directs[index] == max(directs):
            # return "zone: " + zoneStringRepresent(index)
            return index
            break
    # directs = [0] * 8
        # time.sleep(10)
        # start_timer = time.time()
    # cv2.imshow('frame',img)

    
    # Now update the previous frame and previous points
    first_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

# cv2.destroyAllWindows()
# cap.release()


# New main function

# Retrieve the carmera
# cap = cv2.VideoCapture(0)

# Params for ShiTomasi corner detection
# feature_params = dict( maxCorners = 100,
#              qualityLevel = 0.3,
#              minDistance = 7,
#              blockSize = 7 )
# ret, first_frame = cap.read()
# first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
# p0 = cv2.goodFeaturesToTrack(first_gray, mask = None, **feature_params)
# print p0
# time.sleep(1)
# ret, frame = cap.read()

# print directionCalculate(first_frame, frame, p0)
