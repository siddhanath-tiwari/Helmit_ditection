from wsgiref import simple_server
from flask import Flask, request, jsonify
from flask import Response
import os
from flask_cors import CORS
from research.obj import MultiClassObj
from com_in_ineuron_ai_utils.utils import decodeImage

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')


app = Flask(__name__)
CORS(app)


class Api:
    def __init__(self):
        self.filename = "inputImage.jpg"
        modelpath = 'research/ssd_mobilenet_v1_coco_2017_11_17'
        self.objectDetection = MultiClassObj(self.filename, modelpath)