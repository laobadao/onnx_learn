import tensorflow as tf
from tensorflow.python.platform import gfile
import sys

def main(argv):
	model = argv
	graph = tf.get_default_graph()
	graph_def = graph.as_graph_def()
	graph_def.ParseFromString(gfile.FastGFile(model, 'rb').read())
	tf.import_graph_def(graph_def, name='')
	summaryWriter = tf.summary.FileWriter('log/', graph)

if __name__ == '__main__':
	main(sys.argv[1])