"""Script for Public Cloud
@author: A
"""

from flask import Flask
app = Flask(__name__)
import numpy as np
from matplotlib import pyplot as plt

import numpy as np
import cv2 as cv
import time
from main import *
# import tensorflow as tf
from PIL import Image




@app.route('/')
def cv2_im2gray():
    img = cv.imread('shen.jpg')
    # print(img.shape)
    # print(img)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Y = 0.299R + 0.587G + 0.114B

    cv.imwrite('./processed/' + str(time.time()) + ".jpg", gray_img)

if __name__ == "__main__":
    # app.run(debug=False, host='106.52.97.178')
    app.run(debug=True, host='0.0.0.0')