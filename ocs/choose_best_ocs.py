from sklearn.svm import OneClassSVM
import pandas as pd
import os,sys
from sklearn.externals import joblib
import tensorflow as tf, sys
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
from sklearn import metrics

addr = sys.argv[1]
shock_addr = sys.argv[2]
shock_images_path = os.listdir(shock_addr)
shock_len = len([name for name in shock_images_path])
random_addr = sys.argv[3]
random_images_path = os.listdir(random_addr)
random_len = len([name for name in random_images_path])
fileLog = open(sys.argv[5],'w')
oldStdout = sys.stdout
sys.stdout = fileLog
data = pd.read_csv('/nfs/home/pgulyaev/inception_bilabeled_classification/tf_files/'+addr)
testdata = pd.read_csv('/nfs/home/pgulyaev/inception_bilabeled_classification/tf_files/'+sys.argv[4])
y_true = [1.]*shock_len+[-1.]*random_len

for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
    for nu in np.linspace(0.1,0.5,num=5):
        for gamma in [2048.0, 4096.0]:
            res_list = []
            ocs = OneClassSVM(nu=nu, kernel=kernel, gamma=1.0/(gamma))
            ocs.fit(data)
            
            for i in testdata.iterrows():
                res = ocs.predict(i[1])
                res_list.append(res[0])
            
            # y_scores = ocs.decision_function(testdata)        
            y_pred = np.asarray(res_list)
            # precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            print kernel, nu, gamma
            print metrics.accuracy_score(y_true, y_pred),metrics.precision_score(y_true, y_pred), metrics.recall_score(y_true, y_pred) 



sys.stdout = oldStdout

