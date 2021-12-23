# encoding=utf-8
"""Script for Public Cloud
@author: A
"""

from flask import Flask,jsonify
from flask_restful import Resource, Api
from flask import request
import cv2 as cv
from skimage import io
# from io import StringIO
# from easydict import EasyDict
import requests

app = Flask(__name__)
api = Api(app)


@app.route('/',methods=['POST','GET'])

def cv2_im2gray():  # 视图函数
    url = 'http://127.0.0.1'
    # link = request.args.get('link')  # args取get方式参数
    # name = request.args.get('save_name')
    data = {"link": "12345678", "save_name": "101"}
    res = requests.post(url=url, data=data)
    img = io.imread(link)
    # print(img.shape)
    # print(img)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Y = 0.299R + 0.587G + 0.114B
    cv.imwrite('../processed/' + str(name) + ".jpg", gray_img)
    # cv.imwrite('./processed/' + str(time.time()) + ".jpg", gray_img)
    return jsonify(msg="gray image saved")



if __name__ == "__main__":
    # grayflask.run(debug=False, host='106.52.97.178')
    # grayflask.run(debug=True, host='0.0.0.0')
    app.run(host="0.0.0.0", port=5000,debug=True)
