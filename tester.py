import cv2
import os
import numpy as np
import facerecognition as fr


#This module takes images  stored in diskand performs face recognition
test_img=cv2.imread('Testimages/img1.jpg')#test_img path
faces_detected,gray_img=fr.faceDetection(test_img)
print("faces_detected:",faces_detected)


#Comment belows lines when running this program second time.Since it saves training.yml file in directory
faces,faceID=fr.labels_for_training_data('TrainingImages')
recognizer=fr.train_classifier(faces, faceID)
recognizer.save('trainingData.yml')



#Uncomment below line for subsequent runs
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainingData.yml')#use this to load training data for subsequent runs

name={0:"Kiran",1:"Anush"}#creating dictionary containing names for each label

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=recognizer.predict(roi_gray)#predicting the label of given image
    print("confidence:",confidence)
    print("label:",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    if(confidence>0):#If confidence more than 38 then don't print predicted face text on screen
        continue
    fr.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(500,700))
cv2.imshow("face dtecetion tutorial",resized_img)
cv2.waitKey(0)#Waits indefinitely until a key is pressed
cv2.destroyAllWindows






