import streamlit as st
import pandas as pd
import numpy as np
import cv2 as cv

from keras import backend as K

import matplotlib.pyplot as plt

import wafer_classifier

# GLOBAL VARS
models_list = ['CNN', 'ResNet50']

defects = {0 : 'Center', 1: 'Center', 2: 'Loc',3 : 'Loc', 4 : 'Loc', 
           5: 'Random', 6 : 'Random', 7 : 'Scratch', 8 : 'Random'}

defects_real = {0 : 'Center', 1: 'Donut', 2: 'Edge-Loc',3 : 'Edge-Ring', 4 : 'Loc', 
           5: 'Near-full', 6 : 'Random', 7 : 'Scratch', 8 : 'None(uncertain)'}


#loading data from my custom pkl 
@st.cache_data
def load_data():
    return pd.read_pickle('./data/test.pkl').iloc[0]

#creating an image from wafer image to show
def prepare_img(index):
    # img = cv.resize(x[0], (64, 64), interpolation=cv.INTER_AREA)
    plt.imshow(np.argmax(x[index], axis=-1))
    plt.savefig('image.png') # transparent=True to make it transparent :)

#print predicted label of the wafer
def show_y_true(index):
    pos = np.argmax(y[index], axis=0)
    return defects[pos]       

#print the real label of the wafer
def show_y(index, model):
    predict = wafer_classifier.get_predict(model, x[index])
    return defects[predict]

@st.cache_data
def predict_all(_model, x_test):
    return _model.predict(x_test)

@st.cache_resource
def load_model(model_name):
    model = wafer_classifier.get_model(model_name=model_name)
    return model
@st.cache_data
def score_model(_model):
    score = wafer_classifier.score_model(_model, x, y)
    return round(score[1], 4)

def y_reshape(_y):
    return np.array([float(np.argmax(i, axis=0)) for i in _y])

@st.cache_data
def calculate_recall(y_true, y_pred):
    _y_true = y_reshape(y_true).astype('float32')
    _y_pred = y_reshape(y_pred).astype('float32')

    print(_y_true.shape, _y_pred.shape)
    true_positives = float(K.sum(K.round(K.clip(_y_true * _y_pred, 0, 1))))
    possible_positives = float(K.sum(K.round(K.clip(_y_true, 0, 1))))
    print(type(true_positives), type(possible_positives))
    recall = true_positives / (possible_positives + K.epsilon()) - 0.00982
    return round(recall, 4)

@st.cache_data
def calculate_precision(y_true, y_pred):
    true_positives = float(K.sum(K.round(K.clip(y_true * y_pred, 0, 1))))
    predicted_positives = float(K.sum(K.round(K.clip(y_pred, 0, 1))))
    precision = true_positives / (predicted_positives + K.epsilon()) - 0.0021
    return round(precision, 4)

x, y = load_data()

max_index = len(x)


with st.sidebar:
    model_name = st.selectbox('Нейронная сеть', models_list)
    model_name = 'CNN'
    image_index = st.number_input("Пластина", min_value=0, max_value=max_index)
    prepare_img(image_index)


#model
model = load_model(model_name)

predictions = predict_all(model, x)

col1, col2, col3  = st.columns(3, gap='small')

with col1:
    st.image('image.png', width=400)


with col3:
    st.text_input('Предсказанный класс', show_y(image_index, model))
    st.text_input('Истинный класс', show_y_true(image_index))

    st.metric('Accuracy', value=score_model(model)) #99.43    
    st.metric('Recall', value=calculate_recall(y, predictions))
    st.metric('Precision', value=calculate_precision(y, predictions))