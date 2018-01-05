
# coding: utf-8


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from PIL import Image
import glob
import os
import pandas as pd 


IMG_SIZE = 128

for img in glob.glob("train01/*.jpeg"):
    im=cv2.imread(img,0)
    image = Image.open(img)
    #new_im = crop_pic(im,image)
    (h, w) = im.shape[:2]
    center = (w / 2, h / 2)
    
    # cropping the image
    a = int(h/2) - int(IMG_SIZE/2)
    b = int(h/2) + int(IMG_SIZE/2)
    c = int(w/2) - int(IMG_SIZE/2)
    d = int(w/2) + int(IMG_SIZE/2)
    
    #M = cv2.getRotationMatrix2D(center, 180, 1.0)
    #rotated = cv2.warpAffine(image, M, (w, h))

    cropped = im[a:b, c:d]
    
    
    new_name = img.split('.')[0] + "_modify.jpeg"
    #cropped.save(new_name, "JPEG", quality=80, optimize=True, progressive=True)
    #new_im.save(new_name, "JPEG", quality=80, optimize=True, progressive=True)
    cv2.imwrite(new_name,cropped)
    #cv2.imwrite("/Users/freya/Desktop/"+new_name,cropped)
    


