# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2018/5/15$ 12:12$
# @Author  : KOD Chen
# @Email   : 821237536@qq.com
# @File    : model_darknet19$.py
# Description :yolo2网络模型——darknet19.
# --------------------------------------

import os
import tensorflow as tf
import numpy as np
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util
import os.path
import argparse

MODEL_DIR = "yolov2_pb"
MODEL_NAME = "frozen_yolov2.pb"

if not tf.gfile.Exists(MODEL_DIR): #创建目录
    tf.gfile.MakeDirs(MODEL_DIR)

################# 基础层：conv/pool/reorg(带 passthrough 的重组层) #############################################
# 激活函数
def leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.1, name='leaky_relu')  # 或者tf.maximum(0.1*x,x)


# Conv+BN：yolo2中每个卷积层后面都有一个BN层
def conv2d(x, filters_num, filters_size, pad_size=0, stride=1, batch_normalize=True,
           activation=leaky_relu, use_bias=False, name='conv2d'):
    # padding，注意: 不用 padding="SAME",否则可能会导致坐标计算错误
    if pad_size > 0:
        x = tf.pad(x, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
    # 有 BN 层，所以后面有 BN 层的 conv 就不用偏置 bias，并先不经过激活函数 activation
    out = tf.layers.conv2d(x, filters=filters_num, kernel_size=filters_size, strides=stride,
                           padding='VALID', activation=None, use_bias=use_bias, name=name)
    # BN，如果有，应该在卷积层 conv 和激活函数 activation之间
    if batch_normalize:
        out = tf.layers.batch_normalization(out, axis=-1, momentum=0.9, training=False, name=name + '_bn')
    if activation:
        out = activation(out)
    return out


# max_pool
def maxpool(x, size=2, stride=2, name='maxpool'):
    return tf.layers.max_pooling2d(x, pool_size=size, strides=stride)


# reorg layer(带 passthrough (tf.space_to_depth)的重组层)
def reorg(x, stride):
    return tf.space_to_depth(x, block_size=stride)


# 或者return tf.extract_image_patches(x,ksizes=[1,stride,stride,1],strides=[1,stride,stride,1],
# 								rates=[1,1,1,1],padding='VALID')
#########################################################################################################

################################### Darknet19 ###########################################################
# 默认是coco数据集，最后一层维度是 anchor_num * (class_num+5)=5*(80+5)=425
def darknet(images, n_last_channels=425):
    net = conv2d(images, filters_num=32, filters_size=3, pad_size=1, name='conv1')
    net = maxpool(net, size=2, stride=2, name='pool1')

    net = conv2d(net, 64, 3, 1, name='conv2')
    net = maxpool(net, 2, 2, name='pool2')

    net = conv2d(net, 128, 3, 1, name='conv3_1')
    net = conv2d(net, 64, 1, 0, name='conv3_2')
    net = conv2d(net, 128, 3, 1, name='conv3_3')
    net = maxpool(net, 2, 2, name='pool3')

    net = conv2d(net, 256, 3, 1, name='conv4_1')
    net = conv2d(net, 128, 1, 0, name='conv4_2')
    net = conv2d(net, 256, 3, 1, name='conv4_3')
    net = maxpool(net, 2, 2, name='pool4')

    net = conv2d(net, 512, 3, 1, name='conv5_1')
    net = conv2d(net, 256, 1, 0, name='conv5_2')
    net = conv2d(net, 512, 3, 1, name='conv5_3')
    net = conv2d(net, 256, 1, 0, name='conv5_4')
    net = conv2d(net, 512, 3, 1, name='conv5_5')
    shortcut = net  # 存储这一层特征图，以便后面 passthrough 层
    net = maxpool(net, 2, 2, name='pool5')

    net = conv2d(net, 1024, 3, 1, name='conv6_1')
    net = conv2d(net, 512, 1, 0, name='conv6_2')
    net = conv2d(net, 1024, 3, 1, name='conv6_3')
    net = conv2d(net, 512, 1, 0, name='conv6_4')
    net = conv2d(net, 1024, 3, 1, name='conv6_5')

    net = conv2d(net, 1024, 3, 1, name='conv7_1')
    net = conv2d(net, 1024, 3, 1, name='conv7_2')
    # shortcut增加了一个中间卷积层，先采用64个1*1卷积核进行卷积，然后再进行 passthrough 处理
    # 这样26*26*512 -> 26*26*64 -> 13*13*256的特征图
    shortcut = conv2d(shortcut, 64, 1, 0, name='conv_shortcut')
    # tf.space_to_depth() passthrough
    shortcut = reorg(shortcut, 2)
    net = tf.concat([shortcut, net], axis=-1)  # channel整合到一起
    net = conv2d(net, 1024, 3, 1, name='conv8')

    # detection layer:最后用一个1*1卷积去调整channel，该层没有BN层和激活函数
    output = conv2d(net, filters_num=n_last_channels, filters_size=1, batch_normalize=False,
                    activation=None, use_bias=True, name='conv_dec')

    return output


def freeze_graph(model_folder):
    checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径
    print("input_checkpoint:",input_checkpoint)
    output_graph = os.path.join(MODEL_DIR, MODEL_NAME) #PB模型保存路径

    x = tf.random_normal([1, 416, 416, 3], dtype=tf.float32, name="input")

    model_output = darknet(x)

    saver = tf.train.Saver()

    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, ".\\checkpoint_dir\\yolo2_coco.ckpt")
        print(sess.run(model_output).shape)  # (1,13,13,425)

        output_graph_def = graph_util.convert_variables_to_constants(  #模型持久化，将变量值固定
            sess,
            input_graph_def,
            ["conv_dec/BiasAdd"] #如果有多个输出节点，以逗号隔开
        )
        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点

        for op in graph.get_operations():
            print(op.name, op.values())

#########################################################################################################

# if __name__ == '__main__':
#     x = tf.random_normal([1, 416, 416, 3])
#     model_output = darknet(x)
#
#     saver = tf.train.Saver()
#
#     output_graph = "yolov2_pb/yolov2_model.pb"
#     #  output name : 'conv_dec/BiasAdd:0' conv_dec/BiasAdd
#     #  input name : x : 'random_normal:0'
#
#     with tf.Session() as sess:
#
#         # 必须先restore模型才能打印shape;导入模型时，上面每层网络的name不能修改，否则找不到
#         saver.restore(sess, "./checkpoint_dir/yolo2_coco.ckpt")
#         print(sess.run(model_output).shape)  # (1,13,13,425)
#
#         # 得到当前的图的 GraphDef 部分，通过这个部分就可以完成重输入层到输出层的计算过程
#         graph_def = tf.get_default_graph().as_graph_def()
#
#         # 模型持久化，将变量值固定
#         output_graph_def = graph_util.convert_variables_to_constants(
#             sess,
#             graph_def,
#             ["conv_dec/BiasAdd"]  # 需要保存节点的名字
#         )
#         with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
#             f.write(output_graph_def.SerializeToString())  # 序列化输出


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_folder", type=str, help="input ckpt model dir") #命令行解析，help是提示符，type是输入的类型，
    # 这里运行程序时需要带上模型ckpt的路径，不然会报 error: too few arguments
    aggs = parser.parse_args()
    freeze_graph(aggs.model_folder)
# python -m model_darknet19 checkpoint_dir
