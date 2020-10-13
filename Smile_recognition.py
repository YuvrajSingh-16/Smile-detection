import cv2
import urllib.request
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')


def detect(gray, frame):
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for(x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        gray_roi = gray[y:y+w, x:x+h]
        color_roi = frame[y:y+w, x:x+h]
        
        smile = smile_cascade.detectMultiScale(gray_roi, scaleFactor=1.8, minNeighbors=20)
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(color_roi, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
        
    return frame


videoCapture = cv2.VideoCapture(0)
url = "http://192.168.43.1:8080"
# videoCapture.open(url)

while True:
    _, frame = videoCapture.read()
    
    # img_array = np.array(bytearray(urllib.request.urlopen(url).read()), dtype=np.uint8)
    # frame = cv2.imdecode(img_array, -1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    canvas = detect(gray, frame)
    cv2.imshow("IP_webcam", canvas)
    
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break
    
videoCapture.release()
cv2.destroyAllWindows()