
# -*- coding: utf-8 -*-
# Author : Chia Yu

from keras.models import Sequential     
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D
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

# RGB to gray
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def ReadTrainData():
    global img_arr
    global label_train_onehot
    # 讀取檔名
    mypath = '/home/chiayu/Documents/Learn_ML/captcha_recognition/read_test/'
    files = listdir(mypath)
    paths = []
    label = []
    label0 = []
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

def ReadTestData():
    global img_test_arr
    global label_test_onehot
    # 讀取檔名
    mypath = '/home/chiayu/Documents/Learn_ML/captcha_recognition/read_test/'
    files = listdir(mypath)
    paths = []
    label = []
    label0 = []
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

ReadTrainData()

ReadTestData()

# 建立 sequentional 線性堆疊模型，後續只要用 model.add() 方法
model = Sequential()
# 建立模型
model.add(Conv2D(filters=16,kernel_size=(5,5),padding='same',input_shape=(60,160,1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=36, kernel_size=(5,5), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(10, activation='softmax'))
# print(model.summary())

plot_model(model, to_file="model.png", show_shapes = True)


# 定義訓練方式
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 開始訓練
model.fit(x=img_arr, y=label_train_onehot, validation_split=0.2, epochs=100, batch_size=20, verbose=2)
