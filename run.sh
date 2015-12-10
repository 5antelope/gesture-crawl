#!/bin/bash
pip install -r ./requirement.txt
source /usr/local/bin/virtualenvwrapper.sh
mkvirtualenv cv
python test-gesture.py $@
