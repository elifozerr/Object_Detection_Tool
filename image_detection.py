#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import random
import os


#load yolo algorithm
#use opencv dnn module that allows running pre-trained neural networks
#net = cv2.dnn.readNet("Desktop/yolov3_custom_last.weights", "Desktop/yolo_object_detection/yolov3.cfg")
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")


# In[3]:


obj_classes = []
with open("coco.names","r") as f:
    obj_classes = [line.strip() for line in f.readlines()]
    
#print(obj_classes)


# In[4]:


layer_names = net.getLayerNames()

output_layers = []
for i in net.getUnconnectedOutLayers():
    output_layers.append(layer_names[i[0]- 1])


# In[5]:


#load image
#img = cv2.imread("Desktop/high_traffic.jpg")
img = cv2.imread("racket.jpg")
#img = cv2.imread("Desktop/JotForm_project/handbag.jpg")
#img = cv2.imread("Desktop/JotForm_project/suitcase.jpg")
#resize the image
img = cv2.resize(img, None, fx=0.4, fy=0.4)

#keep info of original image
height,width,channels = img.shape


# In[6]:


print(height)
print(width)
print(channels)


# In[7]:


#Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)


# In[8]:


net.setInput(blob)
outs = net.forward(output_layers) #to get final result
#channels
#for b in blob:
#    for n,blob_img in enumerate(b):
#        cv2.imshow(str(n),blob_img)


# In[9]:


#show object information on screen
#to get name of the object using class ids
class_ids = []
#put confident rates
confidences = []
#contain rectangle position info of each object
boxes = []
for out in outs:
    for detection in out :
        #print(detection)
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            #object is detected with high confidence level
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            
            #cv2.circle(image, (center_x,center_y),10,(0,255,0),2)
            #rectangle corordinates
            x = int(center_x - w / 2) #top left x
            y = int(center_y - h / 2) #top left y
            #cv2.circle(image, (center_x,center_y),10,(0,255,0),2)
            #x,y top left --- x + w, y + h = right bottom
            #cv2.rectangle(image, (x,y),(x + w, y + h),(0,255,0),2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

num_of_detected = len(boxes)


# In[10]:


#to show right objects
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#choose font for the image name
font = cv2.FONT_HERSHEY_SIMPLEX
for i in range(num_of_detected):
    if i in indexes:
        x, y, w, h = boxes[i]
        #get the name of the object
        label = str(obj_classes[class_ids[i]])
        
        #obj_counts.insert(class_ids[i], num+1)
        #choose color
        color = (0,255,random.randint(0, 255))
        #x,y top left --- x + w, y + h = right bottom
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        #put object name on image
        cv2.putText(img, label, (x, y + 25), font, 1, color, 2)
        cv2.putText(img, str(confidences[i]), (x+120, y + 25), font, 0.5, (0,0,0), 1)



# In[ ]:


cv2.imshow("Image-Object Detection",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




