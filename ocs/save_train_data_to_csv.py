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
data.to_csv("/nfs/home/pgulyaev/inception_bilabeled_classification/tf_files/"+sys.argv[2], index=False)
