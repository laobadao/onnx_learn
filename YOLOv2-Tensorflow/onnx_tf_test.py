from onnx_tf.frontend import tensorflow_graph_to_onnx_model
import tensorflow as tf

with tf.gfile.GFile("yolov2_pb/frozen_yolov2.pb", "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    onnx_model = tensorflow_graph_to_onnx_model(graph_def,
                                     "conv_dec/BiasAdd",
                                     opset=0)

    file = open("yolov2_pb/onnx_yolov2.onnx", "wb")
    file.write(onnx_model.SerializeToString())
    file.close()