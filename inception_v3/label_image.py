import tensorflow as tf, sys
import os
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn import metrics
# change this as you see fit
# addr = sys.argv[1]
# images_path = os.listdir(addr)
# fileLog = open(sys.argv[2],'w')
# oldStdout = sys.stdout
# sys.stdout = fileLog
shock_addr = sys.argv[1]
shock_images_path = os.listdir(shock_addr)
shock_len = len([name for name in shock_images_path])
random_addr = sys.argv[2]
random_images_path = os.listdir(random_addr)
random_len = len([name for name in random_images_path])
fileLog = open(sys.argv[3],'w')
oldStdout = sys.stdout
sys.stdout = fileLog

y_true = [1.]*shock_len+[-1.]*random_len
y_pred = []
y_scores = []
# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("/nfs/home/pgulyaev/inception_bilabeled_classification/tf_files/digits_retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("/nfs/home/pgulyaev/inception_bilabeled_classification/tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    for i in [1,2]:
        addr = sys.argv[i]
        images_path = os.listdir(addr)
        for image_path in images_path:
            # Read in the image_data
            image_data = tf.gfile.FastGFile(addr+image_path, 'rb').read()
            # Feed the image_data as input to the graph and get first prediction
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            
            predictions = sess.run(softmax_tensor, \
                     {'DecodeJpeg/contents:0': image_data})
            
            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            max_score = 0.0
            pos_score = 0.0
            max_label = ""
            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                if (human_string=='1.0'):
                    pos_score = score
                if (score>max_score):
                    max_label = human_string
                    max_score = score
            # print('%s (score = %.5f)' % (max_label, max_score))
            y_pred.append(float(max_label))
            y_scores.append(pos_score)
print y_pred
print y_scores
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
print precision
print recall
average_precision = average_precision_score(y_true, y_scores)
print 'average_precision ', average_precision
print metrics.precision_score(y_true, y_pred), metrics.recall_score(y_true, y_pred) 
sys.stdout = oldStdout       
  
