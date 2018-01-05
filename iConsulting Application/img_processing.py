import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from PIL import Image
import glob
import os
import pandas as pd

IMG_SIZE = 256

def data_pro(img):
    im = cv2.imread(img, 0)
    (h, w) = im.shape[:2]

    # cropping the image
    a = int(h / 2) - int(IMG_SIZE / 2)
    b = int(h / 2) + int(IMG_SIZE / 2)
    c = int(w / 2) - int(IMG_SIZE / 2)
    d = int(w / 2) + int(IMG_SIZE / 2)

    cropped = im[a:b, c:d]
    return cropped


