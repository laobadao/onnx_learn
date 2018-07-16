# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import cv2
from tensorflow.python.framework import graph_util
import os.path
import argparse

from yolo_v3 import yolo_v3, load_weights, detections_boxes, non_max_suppression

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('input_img', '', 'Input image')
tf.app.flags.DEFINE_string('output_img', '', 'Output image')
tf.app.flags.DEFINE_string('class_names', 'yolo_coco_classes.txt', 'File with class names')
tf.app.flags.DEFINE_string('weights_file', 'yolov3.weights', 'Binary file with detector weights')

tf.app.flags.DEFINE_integer('size', 416, 'Image size')

tf.app.flags.DEFINE_float('conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float('iou_threshold', 0.4, 'IoU threshold')

MODEL_DIR = "yolov3_pb"
MODEL_NAME = "frozen_yolov3.pb"

def load_coco_names(file_name):
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            names[id] = name
    return names


def draw_boxes(boxes, img, cls_names, detection_size):
    draw = ImageDraw.Draw(img)

    for cls, bboxs in boxes.items():
        color = tuple(np.random.randint(0, 256, 3))
        for box, score in bboxs:
            print("score:", score)
            box = convert_to_original_size(box, np.array(detection_size), np.array(img.size))
            print("box:", box)
            draw.rectangle(box, outline=color)
            print("cls_names[cls]:",cls_names[cls])
            draw.text(box[:2], '{} {:.2f}%'.format(cls_names[cls], score * 100), fill=color)


def convert_to_original_size(box, size, original_size):
    ratio = original_size / size
    box = box.reshape(2, 2) * ratio
    return list(box.reshape(-1))


def main(argv=None):
    img = Image.open(FLAGS.input_img)
    img_resized = img.resize(size=(FLAGS.size, FLAGS.size))

    classes = load_coco_names(FLAGS.class_names)

    # placeholder for detector inputs
    inputs = tf.placeholder(tf.float32, [1, FLAGS.size, FLAGS.size, 3], name="input")

    with tf.variable_scope('detector'):
        detections = yolo_v3(inputs, len(classes), data_format='NHWC')
        load_ops = load_weights(tf.global_variables(scope='detector'), FLAGS.weights_file)

    boxes = detections_boxes(detections)

    graph = tf.get_default_graph()
    output_graph = os.path.join(MODEL_DIR, MODEL_NAME)  # PB模型保存路径
    graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        sess.run(load_ops)

        detected_boxes = sess.run(boxes, feed_dict={inputs: [np.array(img_resized, dtype=np.float32)]})

        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess,
            graph_def,
            ["output"]  # 如果有多个输出节点，以逗号隔开
        )

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出

        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点

    filtered_boxes = non_max_suppression(detected_boxes, confidence_threshold=FLAGS.conf_threshold,
                                         iou_threshold=FLAGS.iou_threshold)

    draw_boxes(filtered_boxes, img, classes, (FLAGS.size, FLAGS.size))

    img.save(FLAGS.output_img)
    print("done")
    # img = cv2.imread(FLAGS.output_img, cv2.IMREAD_UNCHANGED)
    # cv2.imshow("image", img)  # 显示图片，后面会讲解
    # cv2.waitKey(0)  # 等待按键

#python ./demo.py --input_img detection_3.jpg --output_img detection_result.jpg

# python log_yolov3.py

if __name__ == '__main__':
    tf.app.run()
