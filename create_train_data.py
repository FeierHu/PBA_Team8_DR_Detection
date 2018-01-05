# importing packages
print ('importing packages...')

import cv2
import numpy as np
import pandas as pd
import os         
from random import shuffle 
from tqdm import tqdm  


# defining some constants
print ('defining some constants...')

TRAIN_DIR = 'train_new/final256'
IMG_SIZE = 256


# loading the labels
print ('loading the labels...')

labels = pd.read_csv('trainLabels.csv')
labels.columns = ['name','class']
print (labels.head())


# extracting label corresponding to image name and on-hot encoding it
print ('extracting label corresponding to image name and on-hot encoding it...')

def create_label(image_name):
    word_label = image_name.split('.')[0]
    for n in range(0,len(labels)):
        if labels['name'][n] == word_label:
            if labels['class'][n] == 0:
                 return np.array([1,0,0,0,0])
            elif labels['class'][n] == 1:
                 return np.array([0,1,0,0,0])
            elif labels['class'][n] == 2:
                 return np.array([0,0,1,0,0])
            elif labels['class'][n] == 3:
                 return np.array([0,0,0,1,0])
            elif labels['class'][n] == 4:
                 return np.array([0,0,0,0,1])

# creating training dataset
print ('creating training dataset...')

training_data = []
for img in tqdm(os.listdir(TRAIN_DIR))
	path = os.path.join(TRAIN_DIR, img)
	img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
	training_data.append([np.array(img_data), create_label(img)])
shuffle(training_data)
np.save('train_data_balanced_256.npy', training_data)
    






