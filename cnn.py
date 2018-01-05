# importing packages
print ('importing packages...')

import cv2
import numpy as np
import pandas as pd
import os         
from random import shuffle 
from tqdm import tqdm      
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


# defining some constants
print ('defining some constants...')

TRAIN_DIR = 'processed_images/train256c'
TEST_DIR = 'test/test256'
IMG_SIZE = 256
LR = 1e-3



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

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR, img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img_data), create_label(img)])
    shuffle(training_data)
    np.save('train_data_balanced_256_c.npy', training_data)
    return training_data


# creating test dataset
print ('creating test dataset...')

def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img_data), img_num])
    shuffle(testing_data)
    np.save('test_data_256.npy', testing_data)
    return testing_data


# if dataset is not created
#train_data = create_train_data()
#test_data = create_test_data()

# if you have already created the dataset
train_data = np.load('train_data_256_c.npy')
test_data = np.load('test_data_256.npy')

# dividing into training and validation set
print ('dividing into training and validation set...')

train = train_data[:-300]
test = train_data[-300:]

X_train = np.array([i[0] for i in train]).reshape([-1,IMG_SIZE, IMG_SIZE, 1])
y_train = [i[1] for i in train]

X_test = np.array([i[0] for i in test]).reshape([-1,IMG_SIZE, IMG_SIZE, 1])
y_test = [i[1] for i in test]

print ('length of train: ', len(train))
print ('length of test: ', len(test))
print ('length of x_train: ', len(X_train))
print ('length of y_train: ', len(y_train))
print ('length of x_test: ', len(X_test))
print ('length of y_test: ', len(y_test))


# building the 6-layer model
print ('building the first model...')

tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 5, activation='softmax')

#adam =  tf.train.AdamOptimizer(learning_rate=LR, beta1=0.99, epsilon=0.1)
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)

model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10, 
          validation_set=({'input': X_test}, {'targets': y_test}), 
         snapshot_step=500, show_metric=True, run_id=MODEL_NAME)






# building the 10-layer model
#print ('building a bigger model...')


#tf.reset_default_graph()

#convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

#convnet = conv_2d(convnet, 32, 5, activation='relu')
#convnet = max_pool_2d(convnet, 5)

#convnet = conv_2d(convnet, 64, 5, activation='relu')
#convnet = max_pool_2d(convnet, 5)

#convnet = conv_2d(convnet, 128, 5, activation='relu')
#convnet = max_pool_2d(convnet, 5)

#convnet = conv_2d(convnet, 64, 5, activation='relu')
#convnet = max_pool_2d(convnet, 5)

#convnet = conv_2d(convnet, 32, 5, activation='relu')
#convnet = max_pool_2d(convnet, 5)

#convnet = fully_connected(convnet, 1024, activation='relu')
#convnet = dropout(convnet, 0.8)

#convnet = fully_connected(convnet, 5, activation='softmax')
#convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

#model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)

#model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10, 
#         validation_set=({'input': X_test}, {'targets': y_test}), 
#          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)




# saving the model

model.save('model_256_c.tflearn')



# loading saved model

# model.load('model_256_c.tflearn')



# testing the model with new pics


d = test_data[0]
img_data, img_num = d

data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
prediction = model.predict([data])[0]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
ax.imshow(img_data, cmap="gray")
print(f"No DR: {prediction[0]}, Mild DR: {prediction[1]}, Moderate DR: {prediction[2]}, Severe DR: {prediction[3]}, Proliferative DR: {prediction[4]}")






