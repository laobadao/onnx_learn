import tensorflow as tf
from tensorflow.python.platform import gfile
model = 'E:\\github\\yolo_v3\\tensorflow-yolo-v3\\yolov3_pb\\frozen_yolov3.pb'
graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
graph_def.ParseFromString(gfile.FastGFile(model, 'rb').read())
tf.import_graph_def(graph_def, name='graph')
summaryWriter = tf.summary.FileWriter('log/', graph)