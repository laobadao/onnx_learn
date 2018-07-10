import tensorflow as tf
from tensorflow.python.platform import gfile
model = 'E:\\Intenginetech\\onnx_model\\YOLOv2-Tensorflow\\yolov2_pb\\frozen_yolov2.pb'
graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
graph_def.ParseFromString(gfile.FastGFile(model, 'rb').read())
tf.import_graph_def(graph_def, name='graph')
summaryWriter = tf.summary.FileWriter('log/', graph)