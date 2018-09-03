import tensorflow as tf
from tensorflow.python.platform import gfile
model = 'frozen_yolov2_space.pb'
graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
graph_def.ParseFromString(gfile.FastGFile(model, 'rb').read())
tf.import_graph_def(graph_def, name='')
summaryWriter = tf.summary.FileWriter('log/', graph)