import tensorflow as tf, sys
import os
from sklearn.svm import OneClassSVM
import pandas as pd
import numpy as np
import os,sys
from sklearn.externals import joblib
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
from sklearn import metrics
shock_addr = sys.argv[1]
shock_images_path = os.listdir(shock_addr)
shock_len = len([name for name in shock_images_path])
random_addr = sys.argv[2]
random_images_path = os.listdir(random_addr)
random_len = len([name for name in random_images_path])
fileLog = open(sys.argv[4],'w')

oldStdout = sys.stdout
sys.stdout = fileLog

y_true = [1.]*shock_len+[-1.]*random_len
res_list = []
ocs = joblib.load('/nfs/home/pgulyaev/inception_bilabeled_classification/tf_files/joblib.pkl') 
testdata = pd.read_csv('/nfs/home/pgulyaev/inception_bilabeled_classification/tf_files/'+sys.argv[3])
res_list = []
for i in testdata.iterrows():
    res = ocs.predict(i[1])
    res_list.append(res[0])

y_scores = ocs.decision_function(testdata)        
y_pred = np.asarray(res_list)
print y_pred
print y_scores
right = 0
for i in xrange(y_pred.size):
    if (y_pred[i]==y_true[i]):
        right+=1
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
print precision
print recall
print 'accuracy ',1.0*right/y_pred.size
print 'f1_score ',f1_score(y_true, y_pred)
average_precision = average_precision_score(y_true, y_scores)
print 'average_precision ', average_precision
print metrics.precision_score(y_true, y_pred), metrics.recall_score(y_true, y_pred) 


sys.stdout = oldStdout

