#encoding:utf8
from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

import keras
from keras.datasets import cifar10
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPool2D
from keras.optimizers import RMSprop,SGD
from keras.models import load_model
from keras.models import model_from_json
import scipy.io as sio

from flask import Flask, render_template, request
from scipy.misc import imsave, imread, imresize
import re
import sys
import os
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications import xception
from keras.applications import inception_v3
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model




# model =  Model(input=base_model.input)
# Model(input=base_model.input,outputs=base_model.get_layer('avg_pool').output)
from keras.models import Model
from keras.layers import Input, Dense
base_model = ResNet50(weights='imagenet',include_top=False)
for layer in base_model.layers:
    layer.trainable = False

base_model_out = base_model.output
my_layer = Flatten()(base_model_out)
my_layer = Dense(256,activation='relu')(my_layer)

# model = Sequential()
# model.add(base_model)
# model.add(Flatten())
# model.add(Dense(256,activation='relu'))

# model.add(Dense(1,activation='sigmoid'))
# model = Sequential() 
# https://stackoverflow.com/questions/43432717/keras-logistic-regression-returns-nan-on-first-epoch
#https://stackoverflow.com/questions/43086548/how-to-manually-specify-class-labels-in-keras-flow-from-directory
# Update: I ended up extending the DirectoryIterator class for the multilabel case. 
# You can now set the attribute "class_mode" to the value "multilabel" 
# and provide a dictionary "multlabel_classes" which maps filenames to their
#  labels. Code: https://github.com/tholor/keras/commit/29ceafca3c4792cb480829c5768510e4bdb489c5
# model.add(Dense(input_dim=1, activation='sigmoid',
#             bias_initializer='normal', units=5))

 
model = Dense(input_dim=1, activation='sigmoid',
            bias_initializer='normal', units=5)(my_layer) 
rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# learning rate should be low, as Resnet is ok
# model.compile(optimizer=rms, loss='categorical_crossentropy')

# model.compile(optimizer='rmsprop',
#http://www.datalearner.com/blog/1051521451493989
#在多标签分类中，大多使用binary_crossentropy损失而不是通常在多类分类中使用
# 的categorical_crossentropy损失函数。这可能看起来不合理，但因为每个输出节点都是独立的
# ，选择二元损失，并将网络输出建模为每个标签独立的bernoulli分布。
model.compile(optimizer=rms ,
            loss='binary_crossentropy',
            metrics=['accuracy'])
for i,layer in enumerate(model.layers):
    print(i,layer.name)

from keras.preprocessing.image import ImageDataGens
train_data_gen = ImageDataGens(
      preprocessing_function=preprocess_input,
      rescale=1./255,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True
)

test_data_gen = ImageDataGens(
      preprocessing_function=preprocess_input,
      rescale=1./255,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True
)
train_dir = original_path = "../../../Dataset/miml-image-data/original/miml_train_data"
test_dir = original_path = "../../../Dataset/miml-image-data/original/miml_test_data"

from keras.preprocessing.image import ImageDataGenerator
classes = ['desert','mountains','sea','sunset','trees']
#https://stackoverflow.com/questions/43086548/how-to-manually-specify-class-labels-in-keras-flow-from-directory
# Update: I ended up extending the DirectoryIterator class for the multilabel case. 
# You can now set the attribute "class_mode" to the value "multilabel" 
# and provide a dictionary "multlabel_classes" which maps filenames to their
#  labels. Code: https://github.com/tholor/keras/commit/29ceafca3c4792cb480829c5768510e4bdb489c5
#keras解决多标签分类问题http://www.datalearner.com/blog/1051521451493989
train_generator = train_data_gen.flow_from_directory(
        train_dir,                                               
        target_size=(150, 150),                                  
        batch_size=20,
        classes =classes,
        class_mode='categorical',
        )                                     
# https://blog.csdn.net/u012193416/article/details/79368855
# 使用flow_from_directory最值得注意的是directory这个参数：
# directory: path to the target directory. It should contain one subdirectory per class. Any PNG, JPG, BMP, PPM or TIF images inside each of the subdirectories directory tree will be included in the generator. 
# 这是官方文档的定义，它的目录格式一定要注意是包含一个子目录下的所有图片这种格式，driectoty路径只要写到标签路径上面的那个路径即可。
# https://blog.csdn.net/weiwei9363/article/details/78635674
test_generator = test_data_gen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        classes = classes,
        class_mode='categorical')

def preprocess_img(img_path):
    img = image.load_img(img_path,target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis = 0)
    x = preprocess_input(x)
    return x
def predict_top_labels(preds,top_num):
    print('Predicted:', decode_predictions(preds, top=top_num)[0])
    return



# history = model.fit_generator(
#       train_generator,
#       steps_per_epoch=100,
#       epochs=30)

# model.save(args.output_model_file)

# a = Input(shape=(32,))
# b = Dense(32)(a)
# model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

# img_path = "../pics/woodgirl.jpg"
# x = preprocess_img(img_path)
# preds = model.predict(x)
# predict_top_labels(preds,3)
# history = model.fit(X, Y_oh, 
#     batch_size=batch_size, 
#     epochs=nb_epoch)

# https://www.cnblogs.com/skyfsm/p/8051705.html
# 读数据
# https://blog.csdn.net/wd1603926823/article/details/52223373
# https://github.com/keras-team/keras/pull/6128
# Overall, I think we should come up with a different design to support multi-label problems. Relying on directories
#  as a labeling mechanism will fundamentally only work neatly for single-label problems.
# 升级keras
# https://github.com/keras-team/keras/blob/master/examples/mnist_tfrecord.py
# https://stackoverflow.com/questions/40111366/writing-tfrecords-with-images-and-multilabels-for-classification
# https://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
# https://tensorflow.google.cn/api_guides/python/reading_data
# https://blog.csdn.net/u013555719/article/details/77899029
# https://blog.csdn.net/u013555719/article/details/77899029
# [TFRecord格式数据]利用TFRecords存储与读取带标签的图片
#标签的格式被称为独热编码(one-hot encoding)这是一种用于多类分类的有标签数据的常见的表示方法.
# Stanford Dogs 数据集之所以被视为多类分类数据,是因为狗会被分类为单一品种,而非多个品种的混合,
# 在现实世界中,当预测狗的品种是,多标签解决方案通常较为有效,因为他们能够同时匹配属于多个品种的狗
# https://blog.csdn.net/chaipp0607/article/details/72960028
# TensorFlow TFRecord数据集的生成与显示
#https://my.oschina.net/u/3800567/blog/1788062
#https://blog.csdn.net/nongfu_spring/article/details/52956763
#我们的方法是这样的，我们将利用网络的卷积层部分，把全连接以上的部分抛掉。然后在我们的训练集和测试集上跑一遍，
#将得到的输出（即“bottleneck feature”，网络在全连接之前的最后一层激活的feature map）记录在两个numpy array里。
# 然后我们基于记录下来的特征训练一个全连接网络。
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html


