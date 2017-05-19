from sklearn.svm import OneClassSVM
import pandas as pd
import os,sys
from sklearn.externals import joblib
addr = sys.argv[1]
data = pd.read_csv('/nfs/home/pgulyaev/inception_bilabeled_classification/tf_files/'+addr)
ocs = OneClassSVM(nu=0.1, kernel="poly", gamma=1.0/(4096.0), coef0=0.1, degree=4)
ocs.fit(data)
joblib.dump(ocs, '/nfs/home/pgulyaev/inception_bilabeled_classification/tf_files/joblib.pkl')

