import tensorflow as tf  # 导入TF库
from tensorflow.keras import layers, optimizers, datasets, Sequential, metrics # 导入TF子库
from tensorflow.keras.models import load_model
import cv2
import os, glob
import random, csv
import matplotlib.pyplot as plt
import numpy as np


img_mean = tf.constant([0.485, 0.456, 0.406])
img_std = tf.constant([0.229, 0.224, 0.225])


def normalize(x, mean=img_mean, std=img_std):
    x = (x - mean)/std
    return x


def preprocess(image_path):
    # x: 图片的路径，y：图片的数字编码
    x = tf.io.read_file(image_path)  # 读入图片
    x = tf.image.decode_jpeg(x, channels=3)  # 将原图解码为通道数为3的三维矩阵
    x = tf.image.resize(x, [244, 244])
    # 数据增强
    # x = tf.image.random_flip_up_down(x) # 上下翻转
    # x = tf.image.random_flip_left_right(x)  # 左右镜像
    x = tf.image.random_crop(x, [224, 224, 3])  # 裁剪
    x = tf.cast(x, dtype=tf.float32) / 255.  # 归一化
    x = normalize(x)
    return x


def recognize_label(label):
    if label == 0:
        img_name = 'Miao_Wa_Zhong_Zi'
    elif label == 1:
        img_name = 'Xiao_Huo_Long'
    elif label == 2:
        img_name = 'Chao_Meng'
    elif label == 3:
        img_name = 'Pi_Ka_Qiu'
    elif label == 4:
        img_name = 'Jie_Ni_Gui'
    else:
        img_name = 'Wrong'
    return img_name


def get_label(image_name):
    x = preprocess(image_name)
    x = tf.reshape(x, (-1, 224, 224, 3))
    out = model(x)
    # print(out)
    label = out.numpy()
    label = np.argmax(label)

    return label


# Load model
model = load_model('Pokemon.h5')

file_name = r'E:\Coding\imrec_exercise\AlexNet\test_pic'
# test_pic_name = r'E:\Coding\imrec_exercise\AlexNet\test_pic\1.png'


for image_name in os.listdir(file_name):    # detect each image
    img_path = file_name + '\\' + image_name
    label = get_label(img_path)
    print(label)
    img_name = recognize_label(label)
    print(img_name)
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (512, 512))
    cv2.imshow(img_name, image)
    cv2.waitKey(0)
    



