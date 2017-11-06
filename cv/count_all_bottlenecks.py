from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import (Input, Dense, Flatten, Dropout)
from keras import applications
from keras.models import Model
import os, glob
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
nb_nonshock_samples = 5150 
nb_border_samples = 100
batch_size = 1

def save_bottlebeck_features():
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input_with_one_dim)
    print(full_path(un_dir['bottleneck_features_dir'])+'border_data.npy')
    # build the VGG16 network
    model = applications.ResNet50(include_top=False, weights='imagenet', input_shape = (img_width, img_height, 3))
    '''
    generator = datagen.flow_from_directory(
        whole_shock_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_shock = model.predict_generator(
        generator, nb_shock_samples)
    np.save(open(full_path(un_dir['bottleneck_features_dir'])+'shock.npy', 'wb'),
            bottleneck_features_shock)
'''
    generator = datagen.flow_from_directory(
        whole_nonshock_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_nonshock = model.predict_generator(
        generator, nb_nonshock_samples)
    np.save(open(full_path(un_dir['bottleneck_features_dir'])+'nonshock.npy', 'wb'),
            bottleneck_features_nonshock)
'''
    generator = datagen.flow_from_directory(
        border_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_border_data = model.predict_generator(
        generator, nb_border_samples)
    np.save(open(full_path(un_dir['bottleneck_features_dir'])+'border_data.npy', 'wb'),
            bottleneck_features_border_data)
'''    
save_bottlebeck_features()
