#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import os
FATHER_DIRECTORY = os.path.dirname(
    os.path.split(os.path.realpath(__file__))[0])
sys.path.append(FATHER_DIRECTORY)
from app import gray

if __name__ == '__main__':
    app.run()
