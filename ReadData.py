import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

img = mpimg.imread('/home/chiayu/Documents/Learn_ML/captcha_recognition/train_test/9504.png')     
print(img.shape)
gray = rgb2gray(img)    
plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.show()

gray_img = np.asarray(gray)
print(gray_img.shape)