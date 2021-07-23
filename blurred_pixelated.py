#!/usr/bin/env python
# coding: utf-8

# In[76]:


import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#READ IMAGE
img = cv2.imread('B:\\Academics\\Projects\\Face Detection\\blur\\3.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#FUNCTION FOR DETECTING BLUR or PIXELATED
#Returns True if either blurred or pixelated
def blur_pixelated(gray_image):
    pix = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    size=60
    thresh=10
    (h, w) = gray_image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    fft = np.fft.fft2(gray_image)
    fftShift = np.fft.fftshift(fft)
    magnitude = 20 * np.log(np.abs(fftShift))
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    #The image will be considered "blurry" if the mean value of the magnitudes is less than the threshold value
   # if pix>400 or mean <=thresh:
    if mean <=thresh:
        print("Picture Rejected")
        print(mean)
        print(pix)
        
        return True
    else:
        print(mean)
        print(pix)
        print("Picture Accepted")
        return False

blur_pixelated(gray)
plt.imshow(gray, cmap="gray")


# In[ ]:




