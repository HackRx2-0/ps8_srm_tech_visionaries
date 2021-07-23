#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import cv2 


def realvscartoon(img):
    s=0
    ym=0
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        auc=s+sum(histr)
        if ym<max(histr):ym=max(histr)
    if auc/ym>20:flag=1
    else :flag=0
    return(flag,auc/ym)


#            output
#real image  1
#cartoon     0


img = cv2.imread('r1.jpeg')
print(realvscartoon(img))


# In[ ]:




