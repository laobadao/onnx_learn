import tensorflow as tf
import os
from tensorflow.python.tools import freeze_graph

slim = tf.contrib.slim
# from tensorflow.contrib.slim.nets import vgg
from tensorflow.contrib.slim.python.slim.nets import resnet_v1

model_path = "/home/nishome/jjzhao/github/tensorflow-onnx/tests/models/resnet_v1_50/resnet_v1_50.ckpt"  


def main():
    tf.reset_default_graph()

    input_node = tf.placeholder(tf.float32, shape=(1, 224, 224, 3), name="input")
    print("input_node:", input_node)
    
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        net, _ = resnet_v1.resnet_v1_50(input_node,1000, is_training=False)
        print("net:", net)
  
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)

        
        tf.train.write_graph(sess.graph_def, './pb_model', 'model.pb')
        
        freeze_graph.freeze_graph('pb_model/model.pb',
                                  '',
                                  False,
                                  model_path,
                                  'resnet_v1_50/logits/BiasAdd',
                                  'save/restore_all',
                                  'save/Const:0',
                                  'pb_model/frozen_resnet_v1_50.pb',
                                  False,
                                  "")

    print("done")


if __name__ == '__main__':
    main()
