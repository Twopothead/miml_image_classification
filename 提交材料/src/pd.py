#encoding:utf8
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.

import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import scipy.io as sio

original_path = "../../../Dataset/miml-image-data/original"
mat_path   = "../../../Dataset/miml-image-data/miml data.mat"
NUM_CLASSES = 5

mat = sio.loadmat(mat_path)
print(mat)
from glob import glob

all_original_images = []
natural_images = []
# natural_images = sorted(glob(original_path+"/*"))
for i in range(1,2001):
    
    natural_images.append(os.path.join(original_path,str(i))+".jpg")
all_original_images = all_original_images + natural_images
all_original_images = pd.DataFrame({'imagepath':all_original_images})
all_original_images['filetype'] = all_original_images.apply(lambda row: row.
imagepath.split(".")[-1],axis=1)

## octave code:
##  load("miml data.mat")
##  length(targets)   
## $ ans =  2000  (targets 5 * 2000 )
# [[ 1  1  1 ..., -1 -1 -1]
#  [-1 -1 -1 ..., -1 -1 -1]
#  [-1 -1 -1 ..., -1 -1 -1]
#  [-1 -1 -1 ..., -1 -1 -1]
#  [-1 -1 -1 ...,  1  1  1]]

mat = sio.loadmat(mat_path)
targets = mat['targets']
class_name = mat['class_name']
bags = mat['bags']

targets = np.array([[elem if elem == 1 else 0 for elem in row]for row in targets])
print(targets)
print(type(targets))

some_hot_list =[]
for i in range(0,2000):
    target_some_hot = [targets[0][i],targets[1][i],targets[2][i],targets[3][i],targets[4][i]]
    some_hot_list.append(target_some_hot)
all_original_images['target_some_hot'] = some_hot_list 

all_original_images.head()
print(all_original_images)
print("There are {} images in MIML image dataset.".format(all_original_images.shape[0])) 
classes =['desert','mountains','sea','sunset','trees']
print(classes)

print(type(all_original_images))
print(type(all_original_images.values[:, :]))
print(all_original_images.values[:, :])
x = []
from skimage import io
from skimage.transform import resize
for path in all_original_images['imagepath']:
    print(path)

#     img = io.imread(path)
#     img = resize(img,(100,100))
#     img = img.transpose()
#     x.append(img)
# x = np.array(x)
# print(x)
# print(type(x))
# targets.transpose()
# print(targets)
# y = targets

# from sklearn.model_selection import train_test_split
# x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=100)

# x_train = x_train.astype('float32')
# x_test  = x_test.astype('float32')

# x_train /= 255
# x_test /= 255
# from __future__ import print_function
# from tqdm import tqdm
# from PIL import ImageFile
# import h5py

# import keras
# from keras.datasets import cifar10
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense,Activation,Flatten,Dropout
# from keras.layers import Conv2D,MaxPool2D
# from keras.optimizers import RMSprop,SGD
# from keras.models import load_model
# from keras.models import model_from_json
# import scipy.io as sio

# from flask import Flask, render_template, request
# from scipy.misc import imsave, imread, imresize
# from keras.applications.inception_v3 import InceptionV3,preprocess_input
# import re
# import sys
# import os
# import matplotlib.pylab as plt
# from keras import backend as K

