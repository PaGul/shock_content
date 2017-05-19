from sklearn.svm import OneClassSVM
import pandas as pd
import os,sys
from sklearn.externals import joblib
addr = sys.argv[1]
images_path = os.listdir(addr)
data = pd.DataFrame()
listdf = []
for image_path in images_path:
    print addr+image_path
    df = pd.read_csv(addr+image_path, header=None, sep=',', engine='python')
    listdf.append(df)
data = pd.concat(listdf)
ocs = OneClassSVM(nu=0.1, kernel="rbf", gamma=2.0/(3.0*4096.0))
ocs.fit(data)
joblib.dump(ocs, '/nfs/home/pgulyaev/inception_bilabeled_classification/tf_files/joblib.pkl') 

