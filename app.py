from flask import Flask,request,send_file,Response,make_response,jsonify
from io import BytesIO
from PIL import Image
import base64
from requests_toolbelt import MultipartEncoder
from body_measurement import bodyMeasureHorizontal
import numpy as np
import cv2
import json
from takeMeasurements import returnChestNWaist
import copy
app = Flask(__name__)
protoFile = "./pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "./pose/mpi/pose_iter_160000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

@app.route('/api/send',methods = ['POST'])
def receive():
    image1 = request.files['image1']
    image2 = request.files['image2']
    height = int(request.form.get('height'))
    frame1 = generateFrame(image1)
    frame2 = generateFrame(image2)
    frame1 = resizeImage(frame1)
    frame2 = resizeImage(frame2)
    horizontalPic,horizontalMeasurements = bodyMeasureHorizontal(copy.copy(frame1),height,net)
    horizontalMeasurements = json.dumps(horizontalMeasurements).encode('utf-8')
    roundMeasure,roundHorizontalPic,roundLateralPic = returnChestNWaist(copy.copy(frame1),frame2,height,net)  
    roundMeasure = json.dumps(roundMeasure).encode('utf-8')
    #response = make_response(png_as_text)
    #response.headers['Content-Type'] = 'image/png'
    #return response


    m = MultipartEncoder(
           fields={'horizaontalPic': ('horizontal.png',generateBase64(horizontalPic), 'image/png'),
                    'horizontalMeasurements':('horizontalMeasurements.json',horizontalMeasurements,'application/JSON'),
                    'horizontalRoundPic':('horizontalRound.png',generateBase64(roundHorizontalPic),'image/png'),
                    'lateralRoundPic':('lateralRound.png',generateBase64(roundLateralPic),'image/png'),
                    'lateralMeasurements':('roundMeasurements.json',roundMeasure,'application/JSON')}
        )
    return Response(m.to_string(), mimetype=m.content_type)

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


if __name__ == "__main__":
    app.run(host = "localhost",port = 5000)