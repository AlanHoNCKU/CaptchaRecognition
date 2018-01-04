# -*- coding: utf-8 -*-
"""
# Author : Chia Yu
"""

import keras as keras 
from keras.models import *
from keras.layers import *
from keras.utils import *
from skimage import data_dir
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array
from os import listdir
from os.path import isfile, isdir, join
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# np.random.seed(20)
from io import BytesIO
from captcha.image import ImageCaptcha
import random
import sys

# RGB to gray
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def ReadTrainData():
    global img_arr
    global label_train_onehot
    global label_train_onehot_1
    global label_train_onehot_2
    global label_train_onehot_3
    # 讀取檔名
    mypath = '/home/chiayu/Documents/Learn_ML/captcha_recognition/TrainData/'
    # mypath = '/home/chiayu/Documents/Learn_ML/captcha_recognition/read_test/'
    files = listdir(mypath)
    paths = []
    label = []
    label0 = []
    label1 = []
    label2 = []
    label3 = []
    img_arr = np.zeros((1, 60, 160))
    i = 0
    for f in files:
        x = f.strip('.png')
        label.append(x)
        paths.append(mypath + x + '.png')
        img = mpimg.imread(paths[i])
        img = rgb2gray(img)
        img = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
        img_arr = np.concatenate((img_arr, img), axis=0)
        i=i+1
    img_arr = img_arr[1:, :, :]
    img_arr = img_arr.reshape(img_arr.shape[0],60,160,1).astype('float32')
    # 取出第一個數字
    for l in range(len(label)):
        label0.append(label[l][0])
    label0 = np.array(label0)
    label_train_onehot = np_utils.to_categorical(label0,num_classes=10)
    # 取出第二個數字
    for l in range(len(label)):
        label1.append(label[l][1])
    label1 = np.array(label1)
    label_train_onehot_1 = np_utils.to_categorical(label1,num_classes=10)
    # 取出第二個數字
    for l in range(len(label)):
        label2.append(label[l][2])
    label2 = np.array(label2)
    label_train_onehot_2 = np_utils.to_categorical(label2,num_classes=10)
    # 取出第三個數字
    for l in range(len(label)):
        label3.append(label[l][3])
    label3 = np.array(label3)
    label_train_onehot_3 = np_utils.to_categorical(label3,num_classes=10)

def ReadTestData():
    global img_test_arr
    global label_test_onehot
    global label_test_onehot_1
    global label_test_onehot_2
    global label_test_onehot_3
    #讀取檔名
    mypath = '/home/chiayu/Documents/Learn_ML/captcha_recognition/TestData/'
    # mypath = '/home/chiayu/Documents/Learn_ML/captcha_recognition/read_test/'
    files = listdir(mypath)
    paths = []
    label = []
    label0 = []
    label1 = []
    label2 = []
    label3 = []
    img_test_arr = np.zeros((1, 60, 160))
    i = 0
    for f in files:
        x = f.strip('.png')
        label.append(x)
        paths.append(mypath + x + '.png')
        img = mpimg.imread(paths[i])
        img = rgb2gray(img)
        img = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
        img_test_arr = np.concatenate((img_test_arr, img), axis=0)
        i=i+1
    img_test_arr = img_test_arr[1:, :, :]
    img_test_arr = img_test_arr.reshape(img_test_arr.shape[0],60,160,1).astype('float32')
    # 取出第一個數字
    for l in range(len(label)):
        label0.append(label[l][0])
    label0 = np.array(label0)
    label_test_onehot = np_utils.to_categorical(label0,num_classes=10)
    # 取出第二個數字
    for l in range(len(label)):
        label1.append(label[l][1])
    label1 = np.array(label1)
    label_test_onehot_1 = np_utils.to_categorical(label1,num_classes=10)
    # 取出第三個數字
    for l in range(len(label)):
        label2.append(label[l][2])
    label2 = np.array(label2)
    label_test_onehot_2 = np_utils.to_categorical(label2,num_classes=10)
    # 取出第四個數字
    for l in range(len(label)):
        label3.append(label[l][3])
    label3 = np.array(label3)
    label_test_onehot_3 = np_utils.to_categorical(label3,num_classes=10)

def gen_img(num):
    num = str(num)
    img = ImageCaptcha()
    img.write (num, '/home/chiayu/Documents/Learn_ML/captcha_recognition/demo_img/'+ num +'.png')

def PredictImage(num):
    gen_img(num)
    x = '/home/chiayu/Documents/Learn_ML/captcha_recognition/demo_img/'+ str(num) +'.png'
    img=mpimg.imread('/home/chiayu/Documents/Learn_ML/captcha_recognition/demo_img/'+ str(num) +'.png')
    x_img = mpimg.imread(x)
    x_img = rgb2gray(x_img) 
    x_img = x_img.reshape(1,60,160,1).astype('float32')  
    # pprint(model.predict(x_img))
    a = model.predict(x_img)
    pred=''
    for i in range(4):
        pred += str(np.argmax(a[i][:]))
    plt.title('Label:'+str(num)+'\n' + 'pred:'+pred)
    plt.imshow(img)
    plt.show()
        
def show_train_history(train_history, train, validation, train1, validation1, train2, validation2, train3, validation3):
     plt.plot(train_history.history[train], 'C3', label='t1')
     plt.plot(train_history.history[train1], 'C3', label='t2')
     plt.plot(train_history.history[train2], 'C3', label='t3')
     plt.plot(train_history.history[train3], 'C3', label='t4')
     plt.plot(train_history.history[validation], 'C4', label='v1')
     plt.plot(train_history.history[validation1], 'C4', label='v2')
     plt.plot(train_history.history[validation2], 'C4', label='v3')
     plt.plot(train_history.history[validation3], 'C4', label='v4')
     plt.title('Train History')
     plt.ylabel(train)
     plt.xlabel('Epoch')
     plt.legend()
     plt.show()


def BuildModel():
    global model
    input_shape=Input(shape=(60,160,1))
    Conv_0 = Conv2D(16, (5,5), padding='same')(input_shape)
    MaxPool_0 = MaxPooling2D((2,2))(Conv_0)
    Conv_1 = Conv2D(36, (5,5), padding='same')(MaxPool_0)
    MaxPool_1 = MaxPooling2D((2,2))(Conv_1)
    Ftn = Flatten()(MaxPool_1)

    fully_cnt0 = Dense(4096, activation='relu', name='fully_cnt0')(Ftn)
    fully_cnt1 = Dense(1024, activation='relu', name='fully_cnt1')(fully_cnt0)

    fully_cnt_0 = Dense(10, activation='softmax', name='fully_cnt_0')(fully_cnt1)
    fully_cnt_1 = Dense(10, activation='softmax', name='fully_cnt_1')(fully_cnt1)
    fully_cnt_2 = Dense(10, activation='softmax', name='fully_cnt_2')(fully_cnt1)
    fully_cnt_3 = Dense(10, activation='softmax', name='fully_cnt_3')(fully_cnt1)

    model = Model(inputs=[input_shape], outputs=[fully_cnt_0,fully_cnt_1,fully_cnt_2,fully_cnt_3])
    # plot_model(model, to_file="model.png", show_shapes = True)

ReadTrainData()

ReadTestData()

BuildModel()


# 繪製架構圖
# print(model.summary())
plot_model(model, to_file="model.png", show_shapes = True)




# 定義訓練方式
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_history = model.fit(img_arr, [label_train_onehot, label_train_onehot_1, label_train_onehot_2, label_train_onehot_3],validation_split=0.2 ,epochs=10, batch_size=100)

# model.evaluate([img_arr], [label_train_onehot, label_train_onehot_1, label_train_onehot_2, label_train_onehot_3], batch_size=500)

scores = model.evaluate([img_test_arr], [label_test_onehot, label_test_onehot_1, label_test_onehot_2, label_test_onehot_3])
print(scores[1])

show_train_history(train_history,'fully_cnt_0_acc', 'val_fully_cnt_0_acc','fully_cnt_1_acc', 'val_fully_cnt_1_acc','fully_cnt_2_acc', 'val_fully_cnt_2_acc', 'fully_cnt_3_acc', 'val_fully_cnt_3_acc')
