# import the necessary packages
import cv2
from tqdm import tqdm  
import os
import pandas as pd

TRAIN_DIR = 'train' 


# loading the labels
print ('loading the labels...')

labels = pd.read_csv('trainLabels.csv')
labels.columns = ['name','class']


# reading an image from the directory and sending it to the appropriate folder
for img in tqdm(os.listdir(TRAIN_DIR)):
	path = os.path.join(TRAIN_DIR, img)
	image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

	word_label = img.split('.')[0]
	for n in range(0,len(labels)):
		if labels['name'][n] == word_label:
			if labels['class'][n] == 0:
				cv2.imwrite("train_new/label0/" + img, image)
			elif labels['class'][n] == 1:
				cv2.imwrite("train_new/label1/" + img, image)
			elif labels['class'][n] == 2:
				cv2.imwrite("train_new/label2/" + img, image)
			elif labels['class'][n] == 3:
				cv2.imwrite("train_new/label3/" + img, image)
			elif labels['class'][n] == 4:
				cv2.imwrite("train_new/label4/" + img, image)
