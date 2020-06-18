import requests
from requests_toolbelt.multipart import decoder
import shutil
import base64
url = 'http://127.0.0.1:5000/api/send'
files = {'image1': open(r'pose1.jpg', 'rb'),'image2':open(r'pose2.jpg', 'rb')}
r = requests.post(url, files=files,data= {'height':180})
testEnrollResponse = r
multipart_data = decoder.MultipartDecoder.from_response(testEnrollResponse)
imageParts = []
dictParts = []
for part in multipart_data.parts:
    print(part.headers)
    if part.headers[b'Content-Type'] == b"application/JSON":
        dictParts.append(part.content)
    else:
        imageParts.append(part.content)

count = 0
for i in imageParts:
    count+=1
#data = base64.b64encode(r.content)
    fh = open(f'bodyMeasure{count}.png','wb')
    fh.write(base64.b64decode(i))
    fh.close()

print(dictParts[0])
print(dictParts[1])
print('Passed')
