#!/usr/bin/python
# -*- coding: UTF-8 -*-
from flask import Flask
from flask_restful import Api,Resource

app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

api.add_resource(HelloWorld, '/')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000,debug=True)
