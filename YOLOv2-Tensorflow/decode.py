# -*- coding: utf-8 -*-

import tensorflow as tf

def decode(model_output, output_sizes=(13, 13), num_class=80, anchors=None):
    '''
     model_output:darknet19网络输出的特征图
     output_sizes:darknet19网络输出的特征图大小，默认是13*13(默认输入416*416，下采样32)
    '''
    H, W = output_sizes
    num_anchors = len(anchors)  # 这里的 anchor 是在configs文件中设置的
    print("num_anchors:", num_anchors)
    anchors = tf.constant(anchors, dtype=tf.float32)  # 将传入的 anchors 转变成 tf 格式的常量列表

    # 13*13*num_anchors*(num_class+5)，第一个维度自适应 batchsize
    print("model_output:", model_output.shape)
    detection_result = tf.reshape(model_output, [-1, H * W, num_anchors, num_class + 5])

    print("detection_result:", detection_result.shape)

    # darknet19 网络输出转化——偏移量、置信度、类别概率
    xy_offset = tf.nn.sigmoid(detection_result[:, :, :, 0:2])  # 中心坐标相对于该cell左上角的偏移量，sigmoid函数归一化到0-1
    wh_offset = tf.exp(detection_result[:, :, :, 2:4])  # 相对于anchor的wh比例，通过e指数解码
    obj_probs = tf.nn.sigmoid(detection_result[:, :, :, 4], name="output_obj")  # 置信度，sigmoid函数归一化到0-1
    print("detection_result[:, :, :, 5:]:", detection_result[:, :, :, 5:].shape)

    class_probs = tf.nn.softmax(detection_result[:, :, :, 5:], name="output_class")  # 网络回归的是'得分',用 softmax 转变成类别概率
    print("softmax class_probs:", class_probs.shape)
    # 构建特征图每个cell的左上角的xy坐标
    height_index = tf.range(H, dtype=tf.float32)  # range(0,13)
    width_index = tf.range(W, dtype=tf.float32)  # range(0,13)
    # 变成x_cell=[[0,1,...,12],...,[0,1,...,12]]和y_cell=[[0,0,...,0],[1,...,1]...,[12,...,12]]
    x_cell, y_cell = tf.meshgrid(height_index, width_index)
    x_cell = tf.reshape(x_cell, [1, -1, 1])  # 和上面[H*W,num_anchors,num_class+5]对应
    print("x_cell:", x_cell.shape)
    y_cell = tf.reshape(y_cell, [1, -1, 1])
    print("y_cell:", y_cell.shape)

    # decode
    bbox_x = (x_cell + xy_offset[:, :, :, 0]) / W
    bbox_y = (y_cell + xy_offset[:, :, :, 1]) / H

    print("anchors[:, 0]:", anchors.shape)
    print("anchors[:, 0]:", anchors[:, 0].shape)
    print("anchors[:, 0]:", anchors[:, 0])

    bbox_w = (anchors[:, 0] * wh_offset[:, :, :, 0]) / W
    bbox_h = (anchors[:, 1] * wh_offset[:, :, :, 1]) / H
    # 中心坐标+宽高 box(x,y,w,h) -> xmin=x-w/2 -> 左上+右下box(xmin,ymin,xmax,ymax)
    bboxes = tf.stack([bbox_x - bbox_w / 2, bbox_y - bbox_h / 2,
                       bbox_x + bbox_w / 2, bbox_y + bbox_h / 2], axis=3)

    print("bboxes.shape:", bboxes.shape)

    bboxes = tf.reshape(bboxes, bboxes.shape, name="output_bboxes")

    return bboxes, obj_probs, class_probs
