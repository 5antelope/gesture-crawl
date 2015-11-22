import numpy as np
import cv2
from math import atan2, degrees, pi


# Calculate the zone of the movement directions
def calcZone(x1, y1, x2, y2):
	zone = -1
	dx = x2 - x1
	dy = y2 - y1
	rads = atan2(dx, dy)
	rads %= 2*pi
	degs = degrees(rads)
	if degs >= 350 or degs <= 10:
		zone = 0
	elif degs > 10 and degs < 80:
		zone = 1
	elif degs >=80 and degs <= 100:
		zone = 2
	elif degs >100 and degs < 170:
		zone = 3
	elif degs >= 170 and degs <= 190:
		zone = 4
	elif degs > 190 and degs < 260:
		zone = 5
	elif degs >= 260 and degs <= 280:
		zone = 6
	elif degs > 280 and degs < 350:
		zone = 7
	return zone


# Retrieve the carmera
cap = cv2.VideoCapture(0)

# Params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
		       qualityLevel = 0.3,
		       minDistance = 7,
		       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
		  maxLevel = 2,
		  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, first_frame = cap.read()
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(first_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(first_frame)

while(True):
	
	# Calculate directions
	directs = [0 for x in range(8)]
	
	ret, frame = cap.read()
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
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
		if zone == -1:
			print 'ERROR CALCULATING THE ZONE'
		else:
			directs[zone] += 1
		mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
		frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
	img = cv2.add(frame,mask)
	for index in range(8):
		if directs[index] == max(directs):
			print "zone: " + str(index)
	cv2.imshow('frame',img)
	k = cv2.waitKey(30) & 0xff

	if k == 'q':
		break
	
	# Now update the previous frame and previous points
	first_gray = frame_gray.copy()
	p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()
