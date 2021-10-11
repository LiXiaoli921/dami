#!/usr/bin/python3
"""Script for Public Cloud
@author: A
"""


import argparse
import numpy as np
from matplotlib import pyplot as plt

import numpy as np
import cv2 as cv
import time
from PIL import Image


# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser(description='命令行传入图像名称')
parser.add_argument('-i','--input', required=True,help="path to input image")
parser.add_argument('-o','--output',required=True,help="path to output image")
# parser.add_argument('--batch-size', type=int, default=32)

# args = parser.parse_args()
args = vars(parser.parse_args())


def cv2_im2gray(image_name):
    image_name = args["input"]
    # solution 1st: to use openCv

    img = cv.imread(image_name)

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Y = 0.299R + 0.587G + 0.114B
    print("cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)结果如下：")
    print('大小:{}'.format(gray_img.shape))
    print("类型：%s" % type(gray_img))
    print(gray_img)

    # cv.imwrite('./processed/'+str(time.time()) + ".jpg", gray_img)
    cv.imwrite('./processed/'+args["output"] + ".jpg", gray_img)



if __name__ == '__main__':
    image_name = args["input"]
    cv2_im2gray(image_name)