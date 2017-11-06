import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import (Input, Dense, Flatten, Dropout)
from keras import applications
from keras.models import Model
import glob
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K
import pandas as pd
from sklearn.svm import OneClassSVM
import os
import sys
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
from sklearn import metrics
import configparser

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

def preprocess_input_with_one_dim(x):
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)[0]
    
un_param = ConfigSectionMap("UniversalParams")
un_dir = ConfigSectionMap("UniversalDirs")
individual_params = ConfigSectionMap("Resnet50Hinge")
mr = ConfigSectionMap("ModelResults")

def mr_full_path(model_result):
    return un_dir["root_dir"] + individual_params["model"] + model_result

def full_path(input_dir):
    return un_dir["root_dir"] + input_dir

# dimensions of our images.
img_width = 299
img_height = 299
whole_shock_dir = full_path(un_dir['whole_shock_dir'])
whole_nonshock_dir = full_path(un_dir['whole_nonshock_dir'])
border_data_dir = full_path(un_dir['border_data_dir'])
nb_shock_samples = 2500
nb_nonshock_samples = 5250 
nb_border_samples = 100
batch_size = 25 
top_model_weights_path = 'bottleneck_fc_modelCV2.h5'
epochs = 100

def train_top_model():
    train_labels = np.array(
       [-1] * 1575 + [1] * 700)
#          [-1] * 2 + [1] * 1)
    validation_labels = np.array(
       [-1] * 525 + [1] * 300)
#          [-1] * 2 + [1] * 1)
    
    shock_data = np.load(open(full_path(un_dir['bottleneck_features_dir'])+'shock.npy', 'rb'))[500:1500]
    nonshock_data = np.load(open(full_path(un_dir['bottleneck_features_dir'])+'nonshock.npy', 'rb'))[1030:3090]
    border_data = np.load(open(full_path(un_dir['bottleneck_features_dir'])+'border_data.npy', 'rb'))[20:60]
    
    shock_data_train = shock_data[0:700]
    shock_data_valid = shock_data[700:]
    nonshock_data_train = nonshock_data[0:1545]
    nonshock_data_valid = nonshock_data[1545:]
    border_data_train = border_data[0:30]
    border_data_valid = border_data[30:]
    
    whole_nonshock_data_train = np.vstack((nonshock_data_train, border_data_train))
    whole_nonshock_data_valid = np.vstack((nonshock_data_valid, border_data_valid))
    
    train_data = np.vstack((whole_nonshock_data_train, shock_data_train))
    validation_data = np.vstack((whole_nonshock_data_valid, shock_data_valid))
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(1, activation='tanh'))
#    model.load_weights(top_model_weights_path)
    model.compile(optimizer='adam',
                  loss='hinge', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)
    
train_top_model()
import gc; gc.collect()
