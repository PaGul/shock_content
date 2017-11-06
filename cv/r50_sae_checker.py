import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import numpy as np
import pandas as pd
import os
import sys
from keras.layers import (Input, Dense, Flatten, Dropout)
from keras.models import Model
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
from sklearn import metrics
import configparser
from keras.regularizers import l2, l1, L1L2

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

def full_path(input_dir):
    return un_dir["root_dir"] + input_dir

un_param = ConfigSectionMap("UniversalParams")
un_dir = ConfigSectionMap("UniversalDirs")
individual_params = ConfigSectionMap("Resnet50Hinge")


class1Name = un_param['class1name']
class2Name = un_param['class2name']
shock_len = 500 
nonshock_len = 1000 

test_activations = pd.read_csv('cv3/activations/testCV3.csv') 
test_activations = test_activations.iloc[:,1:]

print(test_activations.shape)

inputs = Input(shape=(2048,))
h = Dense(8192, activation='relu', activity_regularizer=l1(1e-5))(inputs)
outputs = Dense(2048)(h)

model = Model(input=inputs, output=outputs)
model.load_weights('sparseWeightsCV3.h5')
model.compile(optimizer='adam', loss='mse')



import collections
scores = {}
for i in range(shock_len):
    res = model.evaluate(test_activations.values[i:i+1], test_activations.values[i:i+1], batch_size=1)
    scores[res] = 1
for i in range(shock_len, shock_len + nonshock_len):
    res = model.evaluate(test_activations.values[i:i+1], test_activations.values[i:i+1], batch_size=1)
    scores[res] = 0
oscores = collections.OrderedDict(sorted(scores.items()))
print(oscores)

fileLog = open('cv3SparseRes','w')
oldStdout = sys.stdout
sys.stdout = fileLog

tp = 0
tn = nonshock_len
fp = 0
fn = shock_len
best_threshold = 0.0
f1 = 0.0
best_f1 = 0.0
best_precision = 0.0
best_recall = 0.0
best_acc = 0.0
best_threshold_recall = 0.0
saved_precision = 0.0
saved_recall = 0.0
saved_acc = 0.0
for k, v in oscores.items():
    if (v==1):
        tp+=1
        fn-=1
    if (v==0):
        fp+=1
        tn-=1
    if (tp==0):
        continue
    precision = 1.0 * tp / (tp+fp)
    recall = 1.0 * tp / (tp+fn)
    acc = (tp + tn) / (shock_len + nonshock_len)
    f1 = 2 * precision * recall / (precision + recall)
    if (f1 > best_f1):
        best_f1 = f1
        saved_precision = precision
        saved_recall = recall
        saved_acc = acc
        best_threshold = k
    if (precision > best_precision):
        best_precision = precision
    if (recall > best_recall):
        best_recall = recall
        best_threshold_recall = k
    if (acc > best_acc):
        best_acc = acc
print ('best by f1: ', best_f1, 'threshold', best_threshold, 'precision', saved_precision, 'recall', saved_recall, 'acc', saved_acc)
print ('best precision: ', best_precision)
print ('best recall: ', recall, best_threshold_recall)
print ('best acc: ', best_acc)
import gc; gc.collect()
sys.stdout = oldStdout

