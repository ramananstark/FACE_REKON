import cv2
import os
import sys

cam=cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)

face_detection=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_id = input("\n enter face ID:")


print("\n [INFO] Initialization On Process!!!")

count=0

while(True):

    ret, image = cam.read()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, 1.3, 5)

    for x,y,w,h in faces:
        cv2.rectangle(image, (x,y),(x+w,y+h),(255,0,0),2)
        count+=1

        cv2.imwrite("dataset/User."+str(face_id)+'.'+str(count)+".jpg",gray[y:y+h,x:x+w])
        cv2.imshow("image",image)


    k=cv2.waitKey(100) & 0xff
    if (k==25):
        print("before exit")
        sys.exit()
        print("yea exit")
    elif count>30:
        sys.exit()


print("\n [INFO] Succesfully Executed!!!")
cam.release()
cv2.destroyAllWindows()