import cv2
import numpy as np
import os

cam = cv2.VideoCapture(0)

cl = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

name = input("Enter your Name: ")

frames = []
outputs = []

while True:
    ret, frame = cam.read()
    if ret:
        faces = cl.detectMultiScale(frame)
        for x,y,w,h in faces:
            cut = frame[y:y+h,x:x+h]
            fix = cv2.resize(cut,(100,100))
            gray = cv2.cvtColor(fix, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Cut", gray)

    if cv2.waitKey(1) == ord("c"):
        frames.append(gray.flatten())
        outputs.append([name])

    if cv2.waitKey(1) == ord("q"):
        break



X = np.array(frames)
y = np.array(outputs)

data = np.hstack([y, X])

f_name = "face_data.npy"

if os.path.exists(f_name):
    old = np.load(f_name)
    data = np.vstack([old, data])

np.save(f_name, data)


cam.release()
cv2.destroyAllWindows()
