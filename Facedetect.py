import cv2
import numpy as np
from sklearn.svm import SVC

data = np.load("face_data.npy")

X = data[:,1:].astype(int)
y = data[:,0]

model = SVC(kernel='rbf', random_state = 1)
model.fit(X, y)

cam = cv2.VideoCapture(0)

cl = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


while True:
    ret, frame = cam.read()
    if ret:
        faces = cl.detectMultiScale(frame)
        for x,y,w,h in faces:
            cut = frame[y:y+h,x:x+h]
            fix = cv2.resize(cut,(100,100))
            gray = cv2.cvtColor(fix, cv2.COLOR_BGR2GRAY)

            out = model.predict([gray.flatten()])
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
            cv2.putText(frame, str(out[0]), (x,y-10), cv2.FONT_HERSHEY_PLAIN, 2,(0,0,255), 2)

        cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord("q"):
        break


cam.release()
cv2.destroyAllWindows()
