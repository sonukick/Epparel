from flask import Flask,request,send_file,Response,make_response,jsonify
from io import BytesIO
from PIL import Image
import base64
from body_measurement import bodyMeasureHorizontal
import numpy as np
import cv2
import json
from takeMeasurements import returnChestNWaist
import copy

"""
this sample demonstrates the use of pretrained openpose networks with opencv's dnn module.
//
//  it can be used for body pose detection, using either the COCO model(18 parts):
//  http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel
//  https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/openpose_pose_coco.prototxt
//
//  or the MPI model(16 parts):
//  http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel
//  https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/openpose_pose_mpi_faster_4_stages.prototxt
//
//  (to simplify this sample, the body models are restricted to a single person.)
//
//
//  you can also try the hand pose model:
//  http://posefs1.perception.cs.cmu.edu/OpenPose/models/hand/pose_iter_102000.caffemodel
//  https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/hand/pose_deploy.prototxt
"""

def resizeImage(frame):
    inWidth = frame.shape[1]
    inHeight = frame.shape[0]
    aspect_ratio = inWidth/inHeight
    newHeight = 640
    newWidth = int(newHeight*aspect_ratio)
    dim = (newWidth, newHeight)
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return frame

def generateBase64(image):
    _, buffer = cv2.imencode('.png', image)
    png_as_text = base64.b64encode(buffer)
    return png_as_text

def generateFrame(image):
    frame = np.fromfile(image, np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    return frame

protoFile = "./pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "./pose/mpi/pose_iter_160000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

height = 172
image1 = './20200623_113203.jpg'
image2 = './20200623_113209.jpg'

frame1 = generateFrame(image1)
frame2 = generateFrame(image2)
frame1 = resizeImage(frame1)
frame2 = resizeImage(frame2)
horizontalPic,horizontalMeasurements = bodyMeasureHorizontal(copy.copy(frame1),height,net)
horizontalMeasurements = json.dumps(horizontalMeasurements).encode('utf-8')
roundMeasure,roundHorizontalPic,roundLateralPic = returnChestNWaist(copy.copy(frame1),frame2,height,net)  
roundMeasure = json.dumps(roundMeasure).encode('utf-8')

print(horizontalMeasurements,roundMeasure)

