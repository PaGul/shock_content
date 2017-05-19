import tensorflow as tf, sys
import os
from sklearn.svm import OneClassSVM
import pandas as pd
import numpy as np
import os,sys
from sklearn.externals import joblib
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score

shock_addr = sys.argv[1]
shock_images_path = os.listdir(shock_addr)
shock_len = len([name for name in shock_images_path])
random_addr = sys.argv[2]
random_images_path = os.listdir(random_addr)
random_len = len([name for name in random_images_path])

y_true = [1.]*shock_len+[-1.]*random_len

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("/nfs/home/pgulyaev/inception_bilabeled_classification/tf_files/retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("/nfs/home/pgulyaev/inception_bilabeled_classification/tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

testdata = pd.DataFrame()
listdf = []
with tf.Session() as sess:
    for i in [1,2]:
        addr = sys.argv[i]
        images_path = os.listdir(addr)
        for image_path in images_path:
            image_data = tf.gfile.FastGFile(addr+image_path, 'rb').read()
            bn_tensor = sess.graph.get_tensor_by_name('pool_3/_reshape:0')
        
            bn = sess.run(bn_tensor, \
                     {'DecodeJpeg/contents:0': image_data})
            
            temp = bn[0][:,None]
            temp = np.transpose(temp)
            listdf.append(pd.DataFrame(temp))
testdata = pd.concat(listdf)
testdata.to_csv("/nfs/home/pgulyaev/inception_bilabeled_classification/tf_files/"+sys.argv[3], index=False)
