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

un_param = ConfigSectionMap("Universal")
addr = ConfigSectionMap("Resnet50HingeWithoutAverageLayer")

def preprocess_input_with_one_dim(x):
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)[0]

# dimensions of our images.

img_width = int(un_param['img_width'])
img_height = int(un_param['img_height'])
top_model_weights_path = addr['top_model_weights_path']
train_data_dir = addr['nn_train_data_dir']
validation_data_dir = addr['nn_validation_data_dir']
nb_train_samples = 2275 
nb_validation_samples = 750
epochs = 1000
batch_size = 25
RELEVANT_LAYER_NAME = un_param['relevant_layer_name']

train_path = addr['ocs_train_path']
test_path =  addr['ocs_test_path']
class1Name = un_param['class1name']
class2Name = un_param['class2name']
shock_images_path = os.listdir(test_path+class1Name)
shock_len = len([name for name in shock_images_path if '.jpg' in name])
nonshock_images_path = os.listdir(test_path+class2Name)
nonshock_len = len([name for name in nonshock_images_path if '.jpg' in name])

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    model = ResNet50(weights='imagenet', include_top=False, input_shape = (img_width, img_height,3))
    model = Model(inputs=model.input, outputs=model.layers[-2].output)
    
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples)
    np.save(open('/nfs/home/pgulyaev/ocsvm/resnet50hinge_wal/models/bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples)
    np.save(open('/nfs/home/pgulyaev/ocsvm/resnet50hinge_wal/models/bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open('/nfs/home/pgulyaev/ocsvm/resnet50hinge_wal/models/bottleneck_features_train.npy', 'rb'))
    train_labels = np.array(
        [-1] * 1505 + [1] * 770)

    validation_data = np.load(open('/nfs/home/pgulyaev/ocsvm/resnet50hinge_wal/models/bottleneck_features_validation.npy', 'rb'))
    validation_labels = np.array(
        [-1] * 429 + [1] * 321)
    print(train_data.shape[1:])
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(1, activation='tanh'))
#    model.load_weights(top_model_weights_path)
    model.compile(optimizer='adam',
                  loss='hinge', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)

save_bottlebeck_features()
train_top_model()

model = ResNet50(weights='imagenet', include_top=False, input_shape = (img_width, img_height,3))
model = Model(inputs=model.input, outputs=model.layers[-2].output)

top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(4096, activation='relu', name='relevant_layer'))
top_model.add(Dense(1, activation='tanh'))

top_model.load_weights(top_model_weights_path)

new_model = Sequential()
# for l in model.layers:
#     l.inbound_nodes = []
#     new_model.add(l)
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
    return (x/255.0)[0]


def get_inputs(path):
    files = get_files(path)
    inputs = []
    for i in files:
        x = load_img(i)
        inputs.append(x)
    return inputs

def get_test_inputs(path, class1, class2):
    inputs = []
    inputs.extend(get_inputs(path + class1 + '/*.jpg'))
    inputs.extend(get_inputs(path + class2 + '/*.jpg'))
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

fileLog = open(addr['logname'],'w')
oldStdout = sys.stdout
sys.stdout = fileLog


train_activations = get_activations(new_model, get_inputs(train_path+'/*.jpg'), RELEVANT_LAYER_NAME)
train_activations.to_csv(addr['ocs_train_activations_name'])
test_activations = get_activations(new_model, get_test_inputs(test_path, class1Name, class2Name), RELEVANT_LAYER_NAME)
test_activations.to_csv(addr['ocs_test_activations_name'])
y_true = [1.]*shock_len+[-1.]*nonshock_len
for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
    for nu in np.linspace(0.1,0.9,num=9):
        for gamma in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
            res_list = []
            ocs = OneClassSVM(nu=nu, kernel=kernel, gamma=1.0/(gamma))
            ocs.fit(train_activations)
            
            y_pred = ocs.predict(test_activations)
            y_scores = ocs.decision_function(test_activations)        
            
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            print (kernel, nu, gamma)
            print (metrics.accuracy_score(y_true, y_pred),metrics.precision_score(y_true, y_pred), metrics.recall_score(y_true, y_pred))

import gc; gc.collect()
sys.stdout = oldStdout
