import cv2
import math
import numpy as np




# file = input('nhap ten file hoac duong dan day du: ')
def bodyMeasureHorizontal(frame,height,net):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0) 
    edge = cv2.Canny(blurred,100,200)
    # Specify the input image dimensions
    #resized = cv2.resize(frame, (360, 360))
    dimensions = frame.shape
    inWidth = frame.shape[1]
    inHeight = frame.shape[0]
    print(inHeight,inWidth)
    
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
    head = measurement(body['Head'],body['Neck'])
    shoulder = measurement(body['Right Shoulder'],body['Left Shoulder'])
    right_elbow = measurement(body['Right Shoulder'],body['Right Elbow'])
    left_elbow = measurement(body['Left Shoulder'],body['Left Elbow'])
    right_wrist =  measurement(body['Right Shoulder'],body['Right Elbow']) + measurement(body['Right Elbow'],body['Right Wrist'])
    left_wirst = measurement(body['Left Shoulder'],body['Left Elbow']) + measurement(body['Left Elbow'],body['Left Wrist'])
    hip = measurement(body['Right Hip'],body['Left Hip'])
    right_knee = measurement(body['Right Hip'],body['Right Knee'])
    left_knee = measurement(body['Left Hip'],body['Left Knee'])
    right_ankle = measurement(body['Right Hip'],body['Right Knee']) +  measurement(body['Right Knee'],body['Right Ankle'])
    left_ankle = measurement(body['Left Hip'],body['Left Knee']) + measurement(body['Left Knee'],body['Left Ankle'])
    body['Right Hip'] =[body['Right Hip'][0] -hip/2,body['Right Hip'][1]]
    body['Left Hip'] =[body['Left Hip'][0] +hip/2,body['Left Hip'][1]]
    points[8] = body['Right Hip'] 
    points[11] = body['Left Hip'] 
    chest = body['Neck'][1] + ( body['Chest'][1]-body['Neck'][1])/2
    chest_right = [body['Right Shoulder'][0]+(body['Right Hip'][0]-body['Right Shoulder'][0])/2, chest]
    chest_left = [body['Left Hip'][0]+(body['Left Shoulder'][0]-body['Left Hip'][0])/2, chest]
    eo = body['Chest'][1] + ( body['Right Hip'][1]-body['Chest'][1])/2
    eo_right = [int(body['Right Hip'][0]+(body['Chest'][0]-body['Right Hip'][0])/2),int(eo)]
    eo_left = [int(body['Chest'][0]+(body['Left Hip'][0]-body['Chest'][0])/2),int(eo)]
    print(np.sum(edge[176, 300:310]))
    # print(eo_left)
    while(np.sum(edge[eo_right[0],eo_right[1]-10:eo_right[1]+10]) ==0):
        eo_right[0]-=1
    # while(edge[eo_left[0],eo_left[1]] !=255):
    #     eo_left[0]+=1
    # print(eo_left,eo_right)
    points.append(chest_right)
    points.append(chest_left)
    points.append(eo_right)
    points.append(eo_left)
    print(eo_right,eo_left)
    hip = measurement(body['Right Hip'],body['Left Hip'])
    chest = measurement(chest_right,chest_left)
    # points[14]=[0,0]
    # i=0
    l1=range(19)
    l = [0,1,3,5,6,7,9,10,15]
    for i in l1:
        cv2.circle(frame, (int(points[i][0]), int(points[i][1])), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frame, "{}".format(i), (int(points[i][0]), int(points[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)
        # i+=1
    # high_cm = int(input('nhap chieu cao: '))
    high_cm = height
    fact_high = high_cm - 5

    measurementsOut = {'height':high_cm,
                    'head lenght':distance(head,high,fact_high),
                    'shoulder width':distance(shoulder,high,fact_high),
                    'chest width': distance(chest,high,fact_high),
                    "arm's length ":max([distance(right_elbow,high,fact_high),distance(left_elbow,high,fact_high)]),
                    'hand length ':max([distance(right_wrist,high,fact_high),distance(left_wirst,high,fact_high)]),
                    'hip width' : distance(hip,high,fact_high),
                    'calf length':max([distance(right_knee,high,fact_high),distance(left_knee,high,fact_high)]),
                    'foot length':max([distance(right_ankle,high,fact_high),distance(left_ankle,high,fact_high)])}
    #print('height:',high_cm)
    #print('head lenght:',distance(head,high,fact_high))
    #print('shoulder height:',distance(shoulder,high,fact_high))
    #print('chest width:',distance(chest,high,fact_high))
    #print("arm's length :",max([distance(right_elbow,high,fact_high),distance(left_elbow,high,fact_high)]))
    #print('hand length :',max([distance(right_wrist,high,fact_high),distance(left_wirst,high,fact_high)]))
    #print('hip width:',distance(hip,high,fact_high))
    #print('calf length :',max([distance(right_knee,high,fact_high),distance(left_knee,high,fact_high)]))
    #print('foot length :',max([distance(right_ankle,high,fact_high),distance(left_ankle,high,fact_high)]))

    width = round(frameWidth/frameHeight*480)
    height = 480
    dim = (1024, 720)
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("Output-Keypoints",resized)
    cv2.imwrite('keypoints.jpg', frame) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return frame,measurementsOut
