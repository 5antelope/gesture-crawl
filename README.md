# gesture-crawl

gesture crawl try to capture the gesture of infants/babies and recognize.

after remove backgroud, the gesture on hand will be followed and classified with existing models.

### Run
to start, run `$python gesture.py`

### Set up instructions
set up [opencv](http://www.pyimagesearch.com/2015/06/15/install-opencv-3-0-and-python-2-7-on-osx/) at mac machine

run `$python` under `.virtualenvs/cv/lib/python2.7/site-packages/`, make sure you can `import cv2`

### Gesture dictionary:
- fist [model TWO] move far and close around mouth: **eat**
- five-fingers [model ONE] move to left-top: **airplane**
- five-fingers [model ONE] move left-right: **parents**
- one-finer [model THREE] move up: **up**
