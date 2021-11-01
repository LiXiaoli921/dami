#!/usr/bin/python
# -*- coding: UTF-8 -*-   
import sys
import os
FATHER_DIRECTORY = os.path.dirname(
    os.path.split(os.path.realpath(__file__))[0])
sys.path.append(FATHER_DIRECTORY)

from grayflask.gray import app

if __name__ == '__main__':
    app.run()

