# -*- coding: utf-8 -*-
"""
生成數字資料
# Author : CY
"""
from io import BytesIO
from PIL import Image
from captcha.image import ImageCaptcha
import random
import sys

num_img = sys.argv[1]
num_img = int(num_img)

rand_num = random.sample(range(9999),num_img)
img = ImageCaptcha()
for num in rand_num:
    s = str(num).ljust(4, '0')
    img.write (s, '/home/chiayu/Documents/Learn_ML/captcha_recognition/TestData/'+ s +'.png')
