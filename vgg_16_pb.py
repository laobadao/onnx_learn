import tensorflow as tf
import os
from tensorflow.python.tools import freeze_graph

slim = tf.contrib.slim
from tensorflow.contrib.slim.nets import vgg
model_path = "/home/nishome/jjzhao/repos/vgg16/models/vgg_16.ckpt"  # ����model��·��


def main():
    tf.reset_default_graph()

    input_node = tf.placeholder(tf.float32, shape=(1, 240, 240, 3))
    print("input_node:", input_node)
    #input_node = tf.expand_dims(input_node, 0)
    with slim.arg_scope(vgg.vgg_arg_scope()):
        flow, _ = vgg.vgg_16(input_node)
    flow = tf.cast(flow, tf.uint8, 'out')  # ������������Լ�����Ľӿ����֣�Ϊ��֮��ĵ���pb��ʱ��ʹ��

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)

        # ����ͼ
        tf.train.write_graph(sess.graph_def, './output_model/pb_model', 'model.pb')
        # ��ͼ�Ͳ����ṹһ��
        freeze_graph.freeze_graph('output_model/pb_model/model.pb',
                                  '',
                                  False,
                                  model_path,
                                  'out',
                                  'save/restore_all',
                                  'save/Const:0',
                                  'output_model/pb_model/frozen_model_vgg_16.pb',
                                  False,
                                  "")

    print("done")


if __name__ == '__main__':
    main()