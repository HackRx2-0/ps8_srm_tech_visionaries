#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import cv2
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

img = cv2.imread('B:\\Academics\\Projects\\Face Detection\\Spoof Detection\\sp_3.png')

def calc_hist(img):
    histogram = [0] * 3
    for j in range(3):
        histr = cv2.calcHist([img], [j], None, [256], [0, 256])
        histr *= 255.0 / histr.max()
        histogram[j] = histr
    return np.array(histogram)

def spoof(img):
    
    modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    clf = joblib.load('face_spoofing.pkl')


    sample_number = 1
    count = 0
    measures = np.zeros(sample_number, dtype=np.float)
    

    
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
    
    net.setInput(blob)
    faces3 = net.forward()

    measures[count%sample_number]=0
    height, width = img.shape[:2]
    for i in range(faces3.shape[2]):
        confidence = faces3[0, 0, i, 2]
        if confidence > 0.5:
            box = faces3[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            roi = img[y:y1, x:x1]

            point = (0,0)
            
            img_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
            img_luv = cv2.cvtColor(roi, cv2.COLOR_BGR2LUV)
    
            ycrcb_hist = calc_hist(img_ycrcb)
            luv_hist = calc_hist(img_luv)
    
            feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel())
            feature_vector = feature_vector.reshape(1, len(feature_vector))
    
            prediction = clf.predict_proba(feature_vector)
            prob = prediction[0][1]
    
            measures[count % sample_number] = prob
    
            #cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 0), 2)
    
            point = (x, y-5)
    
            #print (measures, np.mean(measures))
            if 0 not in measures:
                if np.mean(measures) >= 0.8:
                    return True
                else:
                    return False
        
    count+=1


# In[14]:


spoof(img)


# In[ ]:




