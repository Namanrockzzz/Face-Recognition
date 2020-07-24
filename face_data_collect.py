
"""
@author: Naman Taneja
"""

import cv2
import numpy as np

#Initialize Camera
cap=cv2.VideoCapture(0)

#We are going to do face detection using haarcascade
#Face Detection
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

# skip is a count variable for capturing every 10th iage
skip=0

#face_data is an list
face_data=[]

#We want to store tenth face in a particular location ,so create a path variable and we are going to store it in a 6)Face Recognition Project - Generating Selfie Training Data using WebCam folder .It will be an empty folder in my directory .We are going to store the frame in a grayscale image(It should be gray scale image to save memory)
dataset_path='./data/'

face_section = np.zeros((100,100,3)) # Because it was showing errorr that face section is not defined
filename=input("Enter name of the person : ")
while True:
    
    ret,frame=cap.read()
    
    #If due to any reason frame is not captured try it again
    if ret==False:
         continue
    
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#cv2.COLORBGR_2GRAY is color mode#It will convert RGB frame to GRAY frame
    
    faces=face_cascade.detectMultiScale(frame,1.3,5)#This faces will be a list and each face is a tupule
    # print(faces) 
    #It will print all matrices
    
    #We are going to sort faces in order of area
    faces=sorted(faces,key=lambda f:f[2]*f[3])
    #If is faces here [(x,y,w,h),(,,,).....]#indexing is from 0 so we will multiply f[2]*f[3] i.e,w*h
    
    #We are going to make a bounding box around each faces 
    #So we are going to iterate over faces
    # Pick the last face (because it is the largest face acc to area(f[2]*f[3]))
    for face in faces[-1:]:
        x,y,w,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)#Here (0,255,255) is mixture of Green and blue
    
    
        #Extract (Crop out the required face) : Region of Interest
        offset=10 #10 pixels
        #We are going to slice it .
        
        
        #By convention ,In frame [first axis is Y,Second axis is X]
        #So we are adding a padding of 10 pixels in all 4 directions
        face_section= frame[y-offset:y+offset+h,x-offset:x+w+offset]#face_section is some part of frame,frame is an image
        #We are going to resize it in 100cross100 image
        face_section =cv2.resize(face_section,(100,100))
        
        
        #We are going to store every tenth face
        skip+=1
        if(skip%10==0):
            face_data.append(face_section.flatten())
            print(len(face_data))#How many faces I have captured so far
        
    #We are also going to show face_section
                  
    cv2.imshow("Frame",frame)
                  
    cv2.imshow("Face Section",face_section)

    key_pressed=cv2.waitKey(1)&0xFF
    if key_pressed==ord('q'):
        break
    # Convert our face list array into a numpy array
face_data=np.asarray(face_data)#face_data is a python list
print(face_data.shape)
face_data=face_data.reshape((face_data.shape[0],-1))#Number of rows should be same as number of faces,Number of columns will be figured out automatically
print(face_data.shape)
#(7,30000)because we have chosen frame(RGB) 3 for RGB has been multiplied here,if we have chosen gray_frame then it will be(7,10000)
#Save this data into file system

np.save(dataset_path+filename+'.npy',face_data)#filename is user input,.npy is file extension ,face_data is numpy array which we want to save  
print("Data successfully saved at:"+dataset_path+filename+'.npy')
#Data successfully saved at:./data/bh.npy   <---output will be like this if username entered is bh 

cap.release()
cv2.destroyAllWindows()

#If we have more than one person then multiple files will be created in data folder everytime we run the program 
