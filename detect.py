import cv2
from time import sleep

face_classifier = cv2.CascadeClassifier('HaarClassifiers/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('HaarClassifiers/haarcascade_eye.xml')
smile_classifier = cv2.CascadeClassifier('HaarClassifiers/haarcascade_smile.xml')

def face_detector(img):
    #convert image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_classifier.detectMultiScale(roi_gray, 1.3, 5)
        smile = smile_classifier.detectMultiScale(roi_gray, 1.8, 20)
        sleep(0.2)
        for(ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        for (sx, sy, sw, sh) in smile:
           cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0,0,255), 2)
    return img

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow("Face Detector", face_detector(frame))
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()