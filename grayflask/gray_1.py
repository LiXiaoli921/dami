"""Script for Public Cloud
@author: A
"""

from flask import Flask
#from flask_restful import Resource, Api

app = Flask(__name__)
#api = Api(app)


import numpy as np
from matplotlib import pyplot as plt

import numpy as np
import cv2 as cv
import time
from main import *
# import tensorflow as tf
from PIL import Image
from skimage import io
# import urllib2
# import Image
# import cStringIO


# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser(description='命令行传入图像url')
parser.add_argument('-i','--input', required=True,help="path to input image")
parser.add_argument('-o','--output',required=True,help="path to output image")

args = vars(parser.parse_args())

@app.route('/')
def cv2_im2gray():
    # img = cv.imread('shen.jpg')
    img = io.imread(args["input"])
    # print(img.shape)
    # print(img)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Y = 0.299R + 0.587G + 0.114B

    cv.imwrite('../processed/' + str(args["output"]) , gray_img)
    # cv.imwrite('./processed/' + str(time.time()) + ".jpg", gray_img)
    return "gray image saved"

if __name__ == "__main__":
    # app.run(debug=False, host='106.52.97.178')
    # app.run(debug=True, host='0.0.0.0')
    app.run(host="0.0.0.0", port=5000,debug=True)
