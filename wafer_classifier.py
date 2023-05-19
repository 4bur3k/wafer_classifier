import numpy as np
import pandas as pd

import tensorflow as tf
import keras
from keras import layers, Input, models
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

def get_model(model_name):
    if model_name == 'CNN':
        model = create_cnn()
        model.load_weights('./models/cnn/cnn_model')
        return model
    
def get_predict(model, x):
    x = x.reshape((1,32,32,3))
    pred = model.predict(x)
    print(np.argmax(pred, axis=1)[0])
    return np.argmax(pred, axis=1)[0]
    

def create_cnn():
    input_shape = (32, 32, 3)
    input_tensor = Input(input_shape)

    conv_1 = layers.Conv2D(16, (3,3), activation='relu', padding='same')(input_tensor)
    conv_2 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(conv_1)
    conv_3 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(conv_2)

    flat = layers.Flatten()(conv_3)

    dense_1 = layers.Dense(512, activation='relu')(flat)
    dense_2 = layers.Dense(128, activation='relu')(dense_1)
    output_tensor = layers.Dense(9, activation='softmax')(dense_2)

    model = models.Model(input_tensor, output_tensor)
    model.compile(optimizer='Adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

    return model


#fit model again
def refit(model, data):
    pass



#score trained model on validation data
def score_model(model, x_test, y_test):
    return model.evaluate(x_test, y_test)


    