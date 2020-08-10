import cv2
import math
import numpy as np
protoFile = "./pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "./pose/mpi/pose_iter_160000.caffemodel"

def measure_lateral(frame,high_cm,demo,net):
    high_cm -= 5
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    # file = input('nhap ten file hoac duong dan lateral du: ')
    blurred = cv2.GaussianBlur(frame, (5, 5), 0) 
    edge = cv2.Canny(blurred,50,100)
    # Specify the input image dimensions
    dimensions = frame.shape
    inWidth = frame.shape[1]
    inHeight = frame.shape[0]
    
    # Prepare the frame to be fed to the network
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    
    # Set the prepared object as the input blob of the network
    net.setInput(inpBlob)

    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]
    # Empty list to store the detected keypoints
    frameHeight, frameWidth, chanel = frame.shape
    points = []
    for i in range(15):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
    
        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H
        threshold=0
        if prob > threshold : 
            # cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            # cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)
    
            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)
    body={}
    body['Head'] = points[0]
    body['Neck'] = points[1]
    body['Right Shoulder'] = points[2]
    body['Right Elbow'] = points[3]
    body['Right Wrist'] = points[4]
    body['Left Shoulder'] = points[5]
    body['Left Elbow'] = points[6]
    body['Left Wrist'] = points[7]
    body['Right Hip'] = points[8]
    body['Right Knee'] = points[9]
    body['Right Ankle'] = points[10]
    body['Left Hip'] = points[11]
    body['Left Knee'] = points[12]
    body['Left Ankle'] = points[13]
    body['Chest'] = points[14]

    def measurement(a,b):
        return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

    def distance(a,b,x):
        return a/b*x
    high = body['Left Ankle'][1] - body['Head'][1]
    chest = body['Neck'][1] + ( body['Chest'][1]-body['Neck'][1])/2
    chest_right = [int(body['Right Shoulder'][0]+(body['Right Hip'][0]-body['Right Shoulder'][0])/2), int(chest)]
    chest_left = [int(body['Left Hip'][0]+(body['Left Shoulder'][0]-body['Left Hip'][0])/2), int(chest)]
    waist = body['Chest'][1] + (min([ body['Right Hip'][1],body['Left Hip'][1]])-body['Chest'][1])/2
    waist_right = [int(body['Right Hip'][0]+(body['Chest'][0]-body['Right Hip'][0])/2),int(waist)]
    waist_left = [int(body['Chest'][0]+(body['Left Hip'][0]-body['Chest'][0])/2),int(waist)]

    while(np.sum(edge[waist_right[1]-10:waist_right[1]+10,waist_right[0]]) ==0):
        waist_right[0]-=1

    while(np.sum(edge[waist_left[1]-10:waist_left[1]+10,waist_left[0]])==0):
        waist_left[0]+=1
    # print(np.sum(edge[chest_right[1],chest_right[0]]))
    while(np.sum(edge[chest_right[1],chest_right[0]]) ==0):
        chest_right[0]-=1

    while(np.sum(edge[chest_left[1],chest_left[0]])==0):
        chest_left[0]+=1
    hip_right = [body['Right Hip'][0],body['Right Hip'][1]]
    hip_left = [body['Left Hip'][0],body['Left Hip'][1]]
    while(np.sum(edge[hip_right[1],hip_right[0]]) ==0):
        hip_right[0]-=1

    while(np.sum(edge[hip_left[1],hip_left[0]])==0):
        hip_left[0]+=1
    hip_right[1] = hip_left[1]
    # ,points[0],points[13]
    points = [chest_right,chest_left,waist_right,waist_left,hip_right,hip_left]
    waist = measurement(waist_right,waist_left)
    hip = measurement(body['Right Hip'],body['Left Hip'])
    chest = measurement(chest_right,chest_left)
    # ba_ring_horizontal = [shoulder,chest,waist,hip]
    # points[14]=[0,0]
    # i=0
    for i in range(len(points)):
        cv2.circle(frame, (int(points[i][0]), int(points[i][1])), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        # cv2.putText(frame, "{}".format(i), (int(points[i][0]), int(points[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)
        # i+=1
    # high_cm = int(input('nhap chieu cao: '))
    # high_cm = 160
    # fact_high = high_cm - 5
    # print(waist_right,waist_left)
    width = round(frameWidth/frameHeight*480)
    height = 480
    dim = (width, height)
    chest =measurement(chest_right,chest_left) 
    waist = measurement(waist_right,waist_left)
    hip = measurement(hip_right,hip_left)
    lateral_chest = distance(chest,high,high_cm)
    lateral_waist = distance(waist,high,high_cm)
    lateral_hip = distance(hip,high,high_cm)

    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite('keypoints-lateral.jpg', frame)
    cv2.imshow("Output-Keypoints",resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return lateral_chest,lateral_waist,lateral_hip,frame



def measure_horizontal(frame,high_cm,demo,net):
    high_cm -= 5
    
    # file = input('nhap ten file hoac duong dan lateral du: ') 
    blurred = cv2.GaussianBlur(frame, (5, 5), 0) 
    edge = cv2.Canny(blurred,10,20)
    # Specify the input image dimensions
    inWidth = frame.shape[1]
    inHeight = frame.shape[0]
    
    # Prepare the frame to be fed to the network
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    
    # Set the prepared object as the input blob of the network
    net.setInput(inpBlob)

    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]
    # Empty list to store the detected keypoints
    frameHeight, frameWidth, chanel = frame.shape
    points = []
    for i in range(15):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
    
        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H
        threshold=0
        if prob > threshold : 
            # cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            # cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)
    
            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)
    body={}
    body['Head'] = points[0]
    body['Neck'] = points[1]
    body['Right Shoulder'] = points[2]
    body['Right Elbow'] = points[3]
    body['Right Wrist'] = points[4]
    body['Left Shoulder'] = points[5]
    body['Left Elbow'] = points[6]
    body['Left Wrist'] = points[7]
    body['Right Hip'] = points[8]
    body['Right Knee'] = points[9]
    body['Right Ankle'] = points[10]
    body['Left Hip'] = points[11]
    body['Left Knee'] = points[12]
    body['Left Ankle'] = points[13]
    body['Chest'] = points[14]

    def measurement(a,b):
        return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

    def distance(a,b,x):
        return a/b*x


    high = body['Left Ankle'][1] - body['Head'][1]
    hip = measurement(body['Right Hip'],body['Left Hip'])
    body['Right Hip'] =[body['Right Hip'][0] -hip/2,body['Right Hip'][1]]
    body['Left Hip'] =[body['Left Hip'][0] +hip/2,body['Left Hip'][1]]
    hip_right = [int(body['Right Hip'][0]),int(body['Right Hip'][1])]
    hip_left = [int(body['Left Hip'][0]),int(body['Left Hip'][1])]

    while(edge[hip_right[1] , hip_right[0]] ==0):
        hip_right[0]-=1

    while(np.sum(edge[hip_left[1],hip_left[0]])==0):
        hip_left[0]+=1
    hip_right[1] = hip_left[1]
    shoulder_right = list(body['Right Shoulder'])
    shoulder_left = list(body['Left Shoulder'])

    chest = body['Neck'][1] + ( body['Chest'][1]-body['Neck'][1])/2
    chest_right = [int(shoulder_right[0]+(hip_right[0]-shoulder_right[0])/2), int(chest)]
    chest_left = [int(hip_left[0]+(shoulder_left[0]-hip_left[0])/2), int(chest)]

    waist = body['Chest'][1] + (min([ body['Right Hip'][1],body['Left Hip'][1]])-body['Chest'][1])/2
    waist_right = [int(hip_right[0]+(body['Chest'][0]-hip_right[0])/2),int(waist)]
    waist_left = [int(body['Chest'][0]+(hip_left[0]-body['Chest'][0])/2),int(waist)]

    while(np.sum(edge[waist_right[1],waist_right[0]]) ==0):
        waist_right[0]-=1

    while(np.sum(edge[waist_left[1]-3:waist_left[1]+3,waist_left[0]])==0):
        waist_left[0]+=1

    while(np.sum(edge[shoulder_right[1],shoulder_right[0]]) ==0):
        shoulder_right[0]-=1

    while(np.sum(edge[shoulder_left[1]-3:shoulder_left[1]+3,shoulder_left[0]])==0):
        shoulder_left[0]+=1

    # hip_right = [int(shoulder_right[0]),int(body['Right Hip'][1])]
    # hip_left = [int(shoulder_left[0]),int(body['Left Hip'][1])]
    # while(edge[hip_right[1] , hip_right[0]] ==0):
    #     hip_right[0]+=1

    # while(np.sum(edge[hip_left[1],hip_left[0]])==0):
    #     hip_left[0]-=1
    # hip_right[1] = hip_left[1]

    kc_waist = max([ int(body['Chest'][0]) - waist_right[0], waist_left[0] - int(body['Chest'][0]) ])
    waist_right = [body['Chest'][0] - kc_waist,int(waist)]
    waist_left = [body['Chest'][0]+kc_waist,int(waist)]


    # while(np.sum(edge[chest_right[1],chest_right[0]]) ==0):
    #     chest_right[0]-=1

    # while(np.sum(edge[chest_left[1],chest_left[0]])==0):
    #     chest_left[0]+=1

    points = [chest_right,chest_left,waist_right,waist_left,hip_right,hip_left]

    shoulder = measurement(shoulder_right,shoulder_left)
    chest =measurement(chest_right,chest_left)
    # print(chest,high)
    waist = measurement(waist_right,waist_left)
    hip = measurement(hip_right,hip_left)
    horizontal_shoulder = distance(shoulder,high,high_cm)
    horizontal_chest = distance(chest,high,high_cm)
    horizontal_waist = distance(waist,high,high_cm)
    horizontal_hip = distance(hip,high,high_cm)
    # points[14]=[0,0]
    # i=0
    for i in range(len(points)):
        cv2.circle(frame, (int(points[i][0]), int(points[i][1])), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        # cv2.putText(frame, "{}".format(i), (int(points[i][0]), int(points[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)
        # i+=1
    # high_cm = int(input('nhap chieu cao: '))
    # high_cm = 160
    # fact_high = high_cm - 5
    """
    if demo == True:
        width = round(frameWidth/frameHeight*480)
        height = 480
        dim = (width, height)
        resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("Output-Keypoints1",resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    """
    return horizontal_chest,horizontal_waist,horizontal_hip,frame


def returnChestNWaist(frame1,frame2,height,net):
    horizontal_chest,horizontal_waist,horizontal_hip,horizontalPic = measure_horizontal(frame1,height,True,net)
    lateral_chest,lateral_waist,lateral_hip,lateralPic=measure_lateral(frame2,height,True,net)
    return {'horizontal_chest':horizontal_chest,
            'lateral_chest':lateral_chest,
            'horizontal_waist':horizontal_waist,
            'lateral_waist':lateral_waist,
            'horizontal_hip':horizontal_hip,
            'lateral_hip':lateral_hip},horizontalPic,lateralPic