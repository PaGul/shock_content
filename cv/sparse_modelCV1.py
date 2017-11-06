import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
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
from keras.regularizers import l2, l1, L1L2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
RELEVANT_LAYER_NAME = un_param['relevant_layer_name']
class1Name = un_param['class1name']
class2Name = un_param['class2name']

model = applications.ResNet50(weights='imagenet', include_top=False, input_shape = (img_width, img_height,3))

top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(2048, activation='relu', name='relevant_layer'))
top_model.add(Dense(1, activation='tanh'))

top_model.load_weights('/nfs/home/pgulyaev/diploma/cv/bottleneck_fc_modelCV1.h5')

new_model = Sequential()
new_model.add(model)

for l in top_model.layers:
    l.inbound_nodes = []
    new_model.add(l)

def get_files(path):
    files = []
    if os.path.isdir(path):
        files = glob.glob(path + '*.jpg')
    elif path.find('*') > 0:
        files = glob.glob(path)
    else:
        files = [path]

    if not len(files):
        print('No images found by the given path')

    return files

def load_img(img_path):
    img = image.load_img(img_path, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)[0]


def get_inputs(path):
    files = get_files(path)
    inputs = []
    for i in files:
        x = load_img(i)
        inputs.append(x)
    return inputs

def get_activation_function(m, layer):
    x = [m.layers[0].input, K.learning_phase()]
    y = [m.get_layer(layer).output]
    return K.function(x, y)


def get_activations(model, inputs, layer):
    all_activations = []
    activation_function = get_activation_function(model, layer)
    for i in range(len(inputs)):
        activations = activation_function([[inputs[i]], 0])
        all_activations.append(activations[0][0])

    df = pd.DataFrame(all_activations)
    df.reset_index()
    return df

shock_inputs = get_inputs(whole_shock_dir+class1Name+'/'+class1Name+'*')[1000:2000]
nonshock_inputs = get_inputs(whole_nonshock_dir+class2Name+'/'+class2Name+'*')[2060:4120]
nonshock_inputs.extend(get_inputs(border_data_dir+class2Name+'/*')[40:80])

train_activations = get_activations(new_model, shock_inputs[0:900], RELEVANT_LAYER_NAME)
valid_activations = get_activations(new_model, shock_inputs[900:1000], RELEVANT_LAYER_NAME)
test_activations = get_activations(new_model, nonshock_inputs, RELEVANT_LAYER_NAME)



inputs = Input(shape=(2048,))
h = Dense(8192, activation='relu', activity_regularizer=l1(1e-5))(inputs)
outputs = Dense(2048)(h)

model = Model(input=inputs, output=outputs)
model.compile(optimizer='adam', loss='mse')
model.fit(train_activations.values, train_activations.values, 
          validation_data=(valid_activations.values, valid_activations.values),
          batch_size=20, nb_epoch=1000)

model.save_weights('sparseWeightsCV1.h5')

import gc; gc.collect()
                                
