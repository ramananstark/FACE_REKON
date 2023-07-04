import cv2
import numpy
import os
import sys


recoginizer = cv2.face.LBPHFaceRecognizer_create()
recoginizer.read("trainer/trainer.yml")
cascadepath = "haarcascade_frontalface_default.xml"
facecascade = cv2.CascadeClassifier(cascadepath)



font=cv2.FONT_HERSHEY_TRIPLEX

id=0

names=[0,1,2,3,4,5,6,7,8,9]

cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(3,640)
cam.set(4,480)


minW=0.1*cam.get(3)
minH=0.1*cam.get(4)

while True:
    ret, image = cam.read()
    if ret:
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        
        faces=facecascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=9,
            minSize=(int(minW),int(minH)))
        
        for x,y,w,h in faces:
            cv2.rectangle(image, (x,y),(x+w,y+h),(255,0,0),2)
            id,confidence = recoginizer.predict(gray[y:y+h,x:x+w])
            print(id)
            if confidence<100:
                id=names[id]
                confidence = "{0}%".format(round(100-confidence))
            else:
                id="anonymous"
                confidence = "{0}%".format(round(100-confidence))

            cv2.putText(image,str(id),(x+5,y+5),font,1,(255,0,255),2)
            cv2.putText(image,str(confidence),(x+5,y+h-5),font,1,(255,0,255),2)

        cv2.imshow('camera',image)

    k=cv2.waitKey(100) & 0xff
    if (k==25):
        sys.exit()
    


print("\n [INFO] Exicting the process...!!!")
cam.release()
cv2.destroyAllWindows()

