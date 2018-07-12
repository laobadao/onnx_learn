# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2018/5/16$ 17:17$
# @Author  : KOD Chen
# @Email   : 821237536@qq.com
# @File    : Main$.py
# Description :YOLO_v2主函数.
# --------------------------------------

import tensorflow as tf
import cv2

from model_darknet19 import darknet
from decode import decode
from utils import preprocess_image, postprocess, draw_detection
from config import anchors, class_names
from tensorflow.python.framework import graph_util
import os.path
import argparse

MODEL_DIR = "yolov2_pb"
MODEL_NAME = "frozen_yolov2.pb"


def main():
    input_size = (416, 416)
    image_file = 'E:\\Intenginetech\\onnx_model\\YOLOv2-Tensorflow\\yolo2_data\\detection_3.jpg'
    image = cv2.imread(image_file)
    image_shape = image.shape[:2]  # 只取wh，channel=3不取
    print("image_shape:", image_shape)
    # copy、resize416*416、归一化、在第 0 维增加存放 batchsize 维度
    image_cp = preprocess_image(image, input_size)

    # 【1】输入图片进入 darknet19 网络得到特征图，并进行解码得到：xmin xmax表示的边界框、置信度、类别概率
    tf_image = tf.placeholder(tf.float32, [1, input_size[0], input_size[1], 3], name="input")
    model_output = darknet(tf_image)  # darknet19网络输出的特征图
    output_sizes = input_size[0] // 32, input_size[1] // 32  # 特征图尺寸是图片下采样 32 倍
    output_decoded = decode(model_output=model_output, output_sizes=output_sizes,
                            num_class=len(class_names), anchors=anchors)  # 解码

    model_path = "E:\\Intenginetech\\onnx_model\\YOLOv2-Tensorflow\\checkpoint_dir\\yolo2_coco.ckpt"
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        bboxes, obj_probs, class_probs = sess.run(output_decoded, feed_dict={tf_image: image_cp})

    # 【2】筛选解码后的回归边界框——NMS(post process后期处理)
    bboxes, scores, class_max_index = postprocess(bboxes, obj_probs, class_probs, image_shape=image_shape)

    # 【3】绘制筛选后的边界框
    img_detection = draw_detection(image, bboxes, scores, class_max_index, class_names)
    cv2.imwrite(".\\yolo2_data\\detection2.jpg", img_detection)
    print('YOLO_v2 detection has done!')
    cv2.imshow("detection_results", img_detection)
    cv2.waitKey(0)


def freeze_graph(model_folder):
    checkpoint = tf.train.get_checkpoint_state(model_folder)  # 检查目录下ckpt文件状态是否可用
    input_checkpoint = checkpoint.model_checkpoint_path  # 得ckpt文件路径
    print("input_checkpoint:", input_checkpoint)
    output_graph = os.path.join(MODEL_DIR, MODEL_NAME)  # PB模型保存路径


    input_size = (416, 416)
    image_file = 'E:\\Intenginetech\\onnx_model\\YOLOv2-Tensorflow\\yolo2_data\\detection_3.jpg'
    image = cv2.imread(image_file)
    image_shape = image.shape[:2]  # 只取wh，channel=3不取

    # copy、resize416*416、归一化、在第 0 维增加存放 batchsize 维度
    image_cp = preprocess_image(image, input_size)

    # 【1】输入图片进入 darknet19 网络得到特征图，并进行解码得到：xmin xmax表示的边界框、置信度、类别概率
    tf_image = tf.placeholder(tf.float32, [1, input_size[0], input_size[1], 3], name="input")
    model_output = darknet(tf_image)  # darknet19网络输出的特征图
    output_sizes = input_size[0] // 32, input_size[1] // 32  # 特征图尺寸是图片下采样 32 倍
    output_decoded = decode(model_output=model_output, output_sizes=output_sizes,
                            num_class=len(class_names), anchors=anchors)  # 解码

    model_path = "E:\\Intenginetech\\onnx_model\\YOLOv2-Tensorflow\\checkpoint_dir\\yolo2_coco.ckpt"
    saver = tf.train.Saver()

    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, model_path)
        bboxes, obj_probs, class_probs = sess.run(output_decoded, feed_dict={tf_image: image_cp})
        print("bboxes.shape:", bboxes.shape)
        print("obj_probs.shape:", obj_probs.shape)
        print("class_probs.shape:", class_probs.shape)

        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess,
            input_graph_def,
            ["output_bboxes", "output_obj", "output_class"]  # 如果有多个输出节点，以逗号隔开
        )
        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点

        # for op in graph.get_operations():
        #     print(op.name, op.values())


# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("model_folder", type=str, help="input ckpt model dir")  # 命令行解析，help是提示符，type是输入的类型，
#     # 这里运行程序时需要带上模型ckpt的路径，不然会报 error: too few arguments
#     aggs = parser.parse_args()
#     freeze_graph(aggs.model_folder)
##python -m Main checkpoint_dir
#
if __name__ == '__main__':
    main()