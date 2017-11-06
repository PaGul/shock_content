from keras.applications.resnet50 import ResNet50
from keras.layers import (Input, Dense, Flatten, Dropout)
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import sys
import glob 
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K
import pandas as pd
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
import configparser
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
from sklearn import metrics

Config = configparser.ConfigParser()
Config.read('/nfs/home/pgulyaev/config.txt')

def ConfigSectionMap(section):
    dict1 = {}
    options = Config.options(section)
    for option in options:
        try:
            dict1[option] = Config.get(section, option)
            if dict1[option] == -1:
                DebugPrint("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1

un_param = ConfigSectionMap("Universal")
# addr = ConfigSectionMap("Resnet50Test")
addr = ConfigSectionMap("Resnet50Hinge")

def preprocess_input_with_one_dim(x):
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)[0]

img_width = int(un_param['img_width'])
img_height = int(un_param['img_height'])
test_ocs_dir =  addr['ocs_test_path']
class1Name = un_param['class1name']
class2Name = un_param['class2name']
shock_images_path = os.listdir(test_ocs_dir + class1Name)
shock_len = len([name for name in shock_images_path if '.jpg' in name])
nonshock_images_path = os.listdir(test_ocs_dir + class2Name)
nonshock_len = len([name for name in nonshock_images_path if '.jpg' in name])
test_ocs_samples_number = shock_len + nonshock_len
y_true = np.array([-1.]*nonshock_len + [1.]*shock_len).astype(int)
y_true = y_true[:, np.newaxis]

model = ResNet50(weights='imagenet', include_top=False, input_shape = (img_width, img_height,3))

top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(2048, activation='relu', name='relevant_layer'))
top_model.add(Dense(1, activation='tanh'))

top_model.load_weights(addr['top_model_weights_path'])

new_model = Sequential()
new_model.add(model)

for l in top_model.layers:
    l.inbound_nodes = []
    new_model.add(l)


datagen = ImageDataGenerator(preprocessing_function=preprocess_input_with_one_dim)

generator = datagen.flow_from_directory(
    test_ocs_dir,
    target_size=(img_width, img_height),
    batch_size=1,
    class_mode=None,
    shuffle=False)
y_pred = new_model.predict_generator(
    generator, test_ocs_samples_number)
# y_pred = y_pred.astype(int)

fileLog = open('checkResnet50Hinge','w')
oldStdout = sys.stdout
sys.stdout = fileLog
for i in y_pred:
    print (i)
# print (metrics.accuracy_score(y_true, y_pred),metrics.precision_score(y_true, y_pred), metrics.recall_score(y_true, y_pred))

import gc; gc.collect()
sys.stdout = oldStdout
