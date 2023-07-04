import cv2
from PIL import Image
import numpy
import os

path = 'dataset'
recoginizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesandLables(path):

    Imagepaths =[os.path.join(path,i) for i in os.listdir(path)]
    facesamples=[]
    ids=[]

    for i in Imagepaths:
        PIL_img=Image.open(i).convert('L')
        num_img=numpy.array(PIL_img,'uint8')


        id = int(os.path.split(i)[-1].split(".")[1])
        faces=detector.detectMultiScale(num_img)

        for (x,y,w,h) in faces:
            facesamples.append(num_img[y:y+h,x:x+w])
            ids.append(id)


    
    return facesamples,ids

print("\n [INFO] Training under process.........")
faces,ids = getImagesandLables(path)
recoginizer.train(faces,numpy.array(ids))

recoginizer.write("trainer/trainer.yml")

print("\n [INFO] upto {0} faces trained".format(numpy.unique(ids)))
