from flask import Flask, request, jsonify, render_template, make_response
import pickle
import numpy as np
import os
import img_util
import cv2
from sklearn.externals import joblib
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import img_processing

app = Flask(__name__, template_folder="")  # initialise Flask application

#Here can load the saved svm model
#model = joblib.load('svm_model100.pkl')

#make_features function to do the upload image processing to fit the model
def make_features(img_file_path):

    #The features to fit CNN
    IMG_SIZE = 256

    #img_data = cv2.imread(img_file_path, cv2.IMREAD_GRAYSCALE)
    #features = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
    features = cv2.resize(img_file_path, (IMG_SIZE, IMG_SIZE))

    return np.array(features).reshape([-1,IMG_SIZE, IMG_SIZE, 1])

def make_features_svm(img_file_path):
    #this make feature function is makeing features to fit svm model
    image = cv2.imread(img_file_path, 0)
    features = image.ravel()  # 2d -> 1d
    return np.array(features).reshape(1, -1)

# @app.route defines the path and method of the HTTP request which will run
# the `hello()` function
@app.route("/predict", methods=['POST'])
def hello():
    #In packaged img_util is to do saving upload image into a folder.
    #We can save the image into a separate folder named 'images'
    #Then we get the image from particular folder by it path
    image_name = img_util.generate_image_name(request.data)
    image = img_util.get_normalized_image(request.data)
    image.save(img_util.get_save_path(image_name))

    #Here we initialize CNN model
    IMG_SIZE = 256
    LR = 1e-3

    tf.reset_default_graph()

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = fully_connected(convnet, 128, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 5, activation='softmax')

    convnet = regression(convnet, optimizer='adam', loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0,
                        checkpoint_path='model_balanced_256.tflearn')

    #load CNN model
    model.load('model_balanced_256.tflearn')

    #Call make_features function,the return result will be as the feature of CNN
    img = img_util.get_save_path(image_name)
    cropped = img_processing.data_pro(img)
    print(cropped)
    #print(img)
    print(len(cropped))
    #features = make_features(img_util.get_save_path(image_name))
    features = make_features(cropped)

    #Do prediction.
    #The predicted value is the probability of each lable.
    #The label corresponding to different level of Diabetic Retinopathy.
    prediction = model.predict(features)[0]
    #Using argsort to get the max probability, then we will know the level of Diabetic Retinopathy.
    prediction = prediction.argsort()[-1:][::-1]
    #print(prediction)
    #print(type(list(prediction)[0]))

    # To satisfied Json serialization, we transfer the format of value from list to int then to return
    response = {
        "prediction": [int(x) for x in list(prediction)],
    }

    return jsonify(response)


@app.route("/", methods=["GET"])
def form():
    # render_template allows you to dynamically produce HTML pages. These
    # pages are kept in the templates/ directory and you can use the jinja2
    # templating format to take additional args passed to render_template
    # and render them in your HTML page
    return render_template('website.html')


# this if statement means you could create another file and import objects
# from this file, and it wouldn't automatically run the application, but if
# you run this file directly, it does
# useful for reusing code and writing tests to check components work
if __name__ == "__main__":
    # use the run method of the app object to start a local development 
    # server at "<host>:<port>" accessible via your web browser
    app.run(port=9000, host='0.0.0.0')
