# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import tarfile
import time
import tempfile
import requests
import zipfile

import PIL.Image
import numpy as np
import tensorflow as tf
import tf2onnx
import yaml
from tensorflow.core.framework import graph_pb2
from tf2onnx.tfonnx import process_tf_graph
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from PIL import Image, ImageDraw

from object_detection.utils import ops as utils_ops
import cv2
# from decode import decode
from utils import preprocess_image, postprocess, draw_detection, non_max_suppression
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
# from config import anchors, class_names
from scipy import misc
import onnx

TMPPATH = tempfile.mkdtemp()
PERFITER = 1000

IMAGE_SHAPE = (0, 0)
PATH_TO_LABELS = os.path.join('/data1/home/nntool/jjzhao/tensorflow-onnx/tests/data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

RAW_IMAGE_NP = None

PREPROCESSOR_SUB = None

SSD_SQUEEZE = None
SSD_CONCAT_1 = None
TF_SSD_BOXES = None
TF_SSD_BOXES = None
TF_SSD_SCORES = None
RAW_BOX_SCORES = None
RAW_BOX_LOCATIONS = None

def get_beach(inputs):
    """Get beach image as input."""
    for name, shape in inputs.items():
        print("name:", name, "shape:", shape)
        break
    resize_to = shape[1:3]
    print("resize_to:", resize_to)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dog.jpg")
    print("path:", path)
    img = PIL.Image.open(path)
    img = img.resize(resize_to, PIL.Image.ANTIALIAS)
    img_np = np.array(img).astype(np.float32)
    img_np = img_np.reshape(shape)
    # print("img_np:",img_np)
    # vgg16 不需要对图像做处理 其他的需要 
    img_np = img_np / 127.5 - 1
    return {name: img_np}


def get_detection(inputs):
    """Get detection image as input."""
    global IMAGE_SHAPE
    print("get_detection img......")
    for name, shape in inputs.items():
        print("name:", name, "shape:", shape)
        break
    resize_to = shape[1:3]
    print("resize_to:", resize_to)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection_3.jpg")

    image = cv2.imread(path)
    IMAGE_SHAPE = image.shape[:2]  # 只取wh，channel=3不取

    img = PIL.Image.open(path)
    img = img.resize(resize_to, PIL.Image.ANTIALIAS)
    img_np = np.array(img).astype(np.float32)
    img_np = img_np.reshape(shape)
    img_np = img_np / 255
    return {name: img_np}


def get_detection_raw(inputs):
    """Get detection image as input."""
    global IMAGE_SHAPE
    print("get_detection img......")
    for name, shape in inputs.items():
        print("name:", name, "shape:", shape)
        break
    resize_to = shape[1:3]
    print("resize_to:", resize_to)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection_3.jpg")

    image = cv2.imread(path)
    IMAGE_SHAPE = image.shape[:2]  # 只取wh，channel=3不取

    img = PIL.Image.open(path)
    img = img.resize(resize_to, PIL.Image.ANTIALIAS)
    img_np = np.array(img).astype(np.float32)
    img_np = img_np.reshape(shape)
    return {name: img_np}


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def get_ssd_image(inputs):
    """Get ssd image as inputs and preprocessor."""
    global IMAGE_SHAPE
    global RAW_IMAGE_NP
    for name, shape in inputs.items():
        print("name:", name, "shape:", shape)
        break
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection_3.jpg")
    image = Image.open(path)
    print("image_path:", path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    RAW_IMAGE_NP = image_np
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    return {name: image_np_expanded}


def get_random(inputs):
    """Get random input."""
    d = {}
    for k, v in inputs.items():
        d[k] = np.random.sample(v).astype(np.float32)
    return d


def get_random256(inputs):
    """Get random imput between 0 and 255."""
    d = {}
    for k, v in inputs.items():
        d[k] = np.round(np.random.sample(v) * 256).astype(np.float32)
    return d


def get_ramp(inputs):
    """Get ramp input."""
    d = {}
    for k, v in inputs.items():
        size = np.prod(v)
        d[k] = np.linspace(1, size, size).reshape(v).astype(np.float32)
    return d


def load_coco_names(file_name):
    print("load_coco_names")
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            names[id] = name
    return names


def draw_boxes(boxes, img, cls_names, detection_size):
    print("draw_boxes")
    draw = ImageDraw.Draw(img)

    for cls, bboxs in boxes.items():
        color = tuple(np.random.randint(0, 256, 3))
        for box, score in bboxs:
            print("score:", score)
            box = convert_to_original_size(box, np.array(detection_size), np.array(img.size))
            print("box:", box)
            draw.rectangle(box, outline=color)
            print("cls_names[cls]:", cls_names[cls])
            draw.text(box[:2], '{} {:.2f}%'.format(cls_names[cls], score * 100), fill=color)


def convert_to_original_size(box, size, original_size):
    ratio = original_size / size
    box = box.reshape(2, 2) * ratio
    return list(box.reshape(-1))


_INPUT_FUNC_MAPPING = {
    "get_beach": get_beach,
    "get_random": get_random,
    "get_random256": get_random256,
    "get_ramp": get_ramp,
    "get_detection": get_detection,
    "get_detection_raw": get_detection_raw,
    "get_ssd_image": get_ssd_image,
}


def node_name(name):
    """Get node name without io#."""
    assert isinstance(name, str)
    pos = name.find(":")
    if pos >= 0:
        return name[:pos]
    return name


def freeze_session(sess, keep_var_names=None, output_names=None, clear_devices=True):
    """Freezes the state of a session into a pruned computation graph."""
    output_names = [i.replace(":0", "") for i in output_names]
    graph = sess.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(sess, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


class Test(object):
    cache_dir = None

    def __init__(self, url, local, make_input, input_names, output_names, middle_input_names=None, middle_output_names=None,
                 disabled=False, more_inputs=None, rtol=0.01, atol=0.,
                 check_only_shape=False, model_type="frozen", force_input_shape=False):
        self.url = url
        self.make_input = make_input
        self.local = local
        self.input_names = input_names
        self.middle_input_names = middle_input_names
        self.output_names = output_names
        self.middle_output_names = middle_output_names
        self.disabled = disabled
        self.more_inputs = more_inputs
        self.rtol = rtol
        self.atol = atol
        self.check_only_shape = check_only_shape
        self.perf = None
        self.tf_runtime = 0
        self.onnx_runtime = 0
        self.model_type = model_type
        self.force_input_shape = force_input_shape

    def download_file(self):
        """Download file from url."""
        cache_dir = Test.cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        url = self.url
        k = url.rfind('/')
        fname = self.url[k + 1:]
        dir_name = fname + "_dir"
        ftype = None
        if url.endswith(".tar.gz") or url.endswith(".tgz"):
            ftype = 'tgz'
            dir_name = fname.replace(".tar.gz", "").replace(".tgz", "")
        elif url.endswith('.zip'):
            ftype = 'zip'
            dir_name = fname.replace(".zip", "")
        dir_name = os.path.join(cache_dir, dir_name)
        os.makedirs(dir_name, exist_ok=True)
        fpath = os.path.join(dir_name, fname)
        if not os.path.exists(fpath):
            response = requests.get(url)
            if response.status_code not in [200]:
                response.raise_for_status()
            with open(fpath, "wb") as f:
                f.write(response.content)
        model_path = os.path.join(dir_name, self.local)
        if not os.path.exists(model_path):
            if ftype == 'tgz':
                tar = tarfile.open(fpath)
                tar.extractall(dir_name)
                tar.close()
            elif ftype == 'zip':
                zip_ref = zipfile.ZipFile(fpath, 'r')
                zip_ref.extractall(dir_name)
                zip_ref.close()
        return fpath, dir_name

    def show_result(self, result, name):

        if name == "ssd" or name == "ssd_tf_caffe2_post":

            print("--------show_result----------", name, "-----------------")
            label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
            categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                        use_display_name=True)
            category_index = label_map_util.create_category_index(categories)

            vis_util.visualize_boxes_and_labels_on_image_array(
                RAW_IMAGE_NP,
                result['detection_boxes'],
                result['detection_classes'],
                result['detection_scores'],
                category_index,
                instance_masks=result.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=8)

            print("output_dict['detection_boxes']:", result['detection_boxes'].shape)
            print("output_dict['detection_classes']:", result['detection_classes'].shape)
            print("output_dict['detection_scores']:", result['detection_scores'].shape)
            print("RAW_IMAGE_NP:", RAW_IMAGE_NP.shape)

            save_result_img = ""

            if name == "ssd":
                save_result_img = "detection_result_tf.png"
            elif name == "ssd_tf_caffe2_post":
                save_result_img = "detection_result_caffe2.png"

            misc.imsave(save_result_img, RAW_IMAGE_NP)
            print("show result done")

        elif name == "ssd_caffe2":
            print("----------------ssd_caffe2-----------  show result begin ")
            print("ssd caffe2 result[0]:", result[0].shape)
            print("ssd caffe2 result[1]:", result[1].shape)

            result_0_diff = RAW_BOX_LOCATIONS - result[1]
            result_1_diff = RAW_BOX_SCORES - result[0]
            #
            print("result_0_diff:\n", result_0_diff)
            print("result_1_diff:\n", result_1_diff)

        else:
            print(name, "result[0].shape :", result[0].shape)

            if result[0].shape == (1, 1001):
                print(name, " result:", result)
                print(name, " result[0].shape:", result[0].shape)
                index = np.argmax(result[0], axis=1)
                print(name, " index:", index)
            # resnet v1 50 的版本需要 这样去检测 index 它的 shape 维数比较多 1000 类

            if result[0].shape == (1, 1, 1, 1001) or result[0].shape == (1, 1, 1, 1000):
                print(name, " result.shape:", result[0][0][0].shape)
                index = np.argmax(result[0][0][0], axis=1)
                print(name, "\t result[0][0][0][index]:", result[0][0][0][0][index])
                print(name, "\t index:", index)

            # 检测网络 yolov2
            if result[0].shape == (1, 169, 5, 4):
                print(name, " bboxes--result[0].shape:", result[0].shape)
                print(name, " obj--result[1].shape:", result[1].shape)
                print(name, " classes--result[2].shape:", result[2].shape)
                print("len(result):", len(result))
                bboxes, scores, class_max_index = postprocess(result[0], result[1], result[2], image_shape=IMAGE_SHAPE)
                print(name, "\n  postprocess done")
                print(name, " bboxes:", bboxes)
                print(name, " scores:", scores)
                print(name, " class_max_index:", class_max_index)

            if result[0].shape == (1, 10647, 85):
                path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection_3.jpg")
                img = Image.open(path)

                classes = load_coco_names("yolo_coco_classes.txt")

                filtered_boxes = non_max_suppression(result[0], confidence_threshold=0.5,
                                                     iou_threshold=0.4)

                draw_boxes(filtered_boxes, img, classes, (416, 416))

                img.save("detection_result.jpg")

    def run_tensorflow(self, sess, inputs):
        print('run_tensorflow(): so we have a reference output')
        """Run model on tensorflow so we have a referecne output."""
        feed_dict = {}
        for k, v in inputs.items():
            k = sess.graph.get_tensor_by_name(k)
            feed_dict[k] = v
        result = sess.run(self.output_names, feed_dict=feed_dict)
        # print("run_tensorflow result:", result)
        self.show_result(result, "tensorflow")

        if self.perf:
            start = time.time()
            for _ in range(PERFITER):
                _ = sess.run(self.output_names, feed_dict=feed_dict)
            self.tf_runtime = time.time() - start
        return result

    def ssd_tensorflow_pre(self, inputs):
        for name, data in inputs.items():
            #print("name:", name, "data:", data)
            break

        image_value = inputs[name]
        print("image_value:", image_value.shape)
        ops = tf.get_default_graph().get_operations()
        print("len ops:", len(ops))
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        print("len(all_tensor_names):", len(all_tensor_names))

        tensor_dict = {}

        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

        print("tensor_dict:", len(tensor_dict))

        if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image_value.shape[0], image_value.shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)

        image_tensor = tf.get_default_graph().get_tensor_by_name(name)
        print("image_tensor:", image_tensor)
        print("tensor_dict:", len(tensor_dict))

        return tensor_dict, image_tensor, image_value

    def run_tensorflow_ssd(self, sess, inputs):
        print("----" * 20)
        print('run_tensorflow_ssd(): so we have a reference output')
        """Run model on tensorflow so we have a referecne output."""

        tensor_dict, image_tensor, image_value = self.ssd_tensorflow_pre(inputs)

        # run inference
        result = sess.run(tensor_dict, feed_dict={image_tensor: image_value})

        global TF_SSD_BOXES
        global TF_SSD_SCORES

        TF_SSD_BOXES = result['detection_boxes'][0]
        TF_SSD_SCORES = result['detection_scores'][0]

        result['num_detections'] = int(result['num_detections'][0])
        result['detection_classes'] = result[
            'detection_classes'][0].astype(np.uint8)
        result['detection_boxes'] = result['detection_boxes'][0]
        result['detection_scores'] = result['detection_scores'][0]
        if 'detection_masks' in result:
            result['detection_masks'] = result['detection_masks'][0]

        self.show_result(result, "ssd")

        # print("result:", result)
        # 'Squeeze': [1, 1917, 4],
        # 'concat_1': [1, 1917, 91],

        for name, shape in self.middle_input_names.items():
            print("middle input name:", name, "shape:", shape)
            break

        middle_tensor_name = name

        # 'Postprocessor/raw_box_scores:0'
        raw_box_scores_name = self.middle_output_names[0]
        raw_box_locations_name = self.middle_output_names[1]

        preprocessor_tensor = tf.get_default_graph().get_tensor_by_name(middle_tensor_name)
        raw_box_scores_tensor = tf.get_default_graph().get_tensor_by_name(raw_box_scores_name)
        raw_box_locations_tensor = tf.get_default_graph().get_tensor_by_name(raw_box_locations_name)

        print("preprocessor_tensor:", preprocessor_tensor)
        print("raw_box_scores_tensor:", raw_box_scores_tensor)
        print("raw_box_locations_tensor:", raw_box_locations_tensor)

        sess.run(tf.global_variables_initializer())

        global PREPROCESSOR_SUB
        global SSD_SQUEEZE
        global SSD_CONCAT_1
        global RAW_BOX_SCORES
        global RAW_BOX_LOCATIONS

        middle_output = sess.run(middle_tensor_name, feed_dict={image_tensor: image_value})
        raw_box_scores_output = sess.run(raw_box_scores_name, feed_dict={image_tensor: image_value})
        raw_box_locations_output = sess.run(raw_box_locations_name, feed_dict={image_tensor: image_value})

        PREPROCESSOR_SUB = middle_output
        print("middle_input", middle_output.shape)
        print("raw_box_locations_output", raw_box_locations_output.shape)
        print("raw_box_scores_output", raw_box_scores_output.shape)

        RAW_BOX_LOCATIONS = raw_box_locations_output
        RAW_BOX_SCORES = raw_box_scores_output

        print("run tensorflow  ssd done ")

        result = raw_box_scores_output, raw_box_locations_output

        return result

    def run_tf_ssd_post(self, g , inputs):

        for name, data in inputs.items():
            print("run_tf_ssd_post name:", name, "---data----:", data.shape)

        print("----" * 20)

        with tf.Session(graph=g) as sess:
            # TODO caffe2 ssd 输出结果 作为输入 用 tensorflow 进行后处理 不能传 imagetensor
            # 要传  post 需要的 4 个 inputs 其中两个为 caffe2 的输出

            tensor_dict, image_tensor, image_value = self.ssd_tensorflow_pre(inputs)

            # "image_tensor:0"

            image_tensor_name = name
            raw_box_scores_name = self.middle_output_names[0]
            raw_box_locations_name = self.middle_output_names[1]

            stack_1_name = "Postprocessor/stack_1:0"

            image_tensor = tf.get_default_graph().get_tensor_by_name(image_tensor_name)

            stack_1_tensor = tf.get_default_graph().get_tensor_by_name(stack_1_name)
            raw_box_scores_tensor = tf.get_default_graph().get_tensor_by_name(raw_box_scores_name)
            raw_box_locations_tensor = tf.get_default_graph().get_tensor_by_name(raw_box_locations_name)
            print("stack_1_tensor:", stack_1_tensor)

            sess.run(tf.global_variables_initializer())

            stack_1_result = sess.run(stack_1_tensor, feed_dict={image_tensor:inputs[name]})
            print("stack_1_result:", stack_1_result.shape)

            feed_dict = {stack_1_tensor: stack_1_result, raw_box_scores_tensor: RAW_BOX_SCORES, raw_box_locations_tensor: RAW_BOX_LOCATIONS}

            result = sess.run(tensor_dict, feed_dict=feed_dict)

            # print("TF_SSD_BOXES:", TF_SSD_BOXES)
            # print("TF_SSD_SCORES:", TF_SSD_SCORES)
            # print("result['detection_boxes'][0]:", result['detection_boxes'][0])
            # print("result['detection_scores'][0]:", result['detection_scores'][0])
            #
            # print("TF_SSD_BOXES sub\n:", TF_SSD_BOXES - result['detection_boxes'][0])
            # print("TF_SSD_SCORES sub\n:", TF_SSD_SCORES - result['detection_scores'][0])

            result['num_detections'] = int(result['num_detections'][0])
            result['detection_classes'] = result[
                'detection_classes'][0].astype(np.uint8)
            result['detection_boxes'] = result['detection_boxes'][0]
            result['detection_scores'] = result['detection_scores'][0]
            if 'detection_masks' in result:
                result['detection_masks'] = result['detection_masks'][0]

            self.show_result(result, "ssd_tf_caffe2_post")

        return result

    @staticmethod
    def to_onnx(tf_graph, opset=None, shape_override=None):
        """Convert graph to tensorflow."""
        return process_tf_graph(tf_graph, continue_on_error=False, opset=opset, shape_override=shape_override)

    def run_caffe2(self, name, onnx_graph, inputs):
        """Run test again caffe2 backend."""
        print("---------------------------------------------run_caffe2()---------------------------------------------")
        import caffe2.python.onnx.backend

        if self.middle_output_names:
            model_proto = onnx_graph.make_model("test", inputs.keys(), self.middle_output_names)
        else:
            model_proto = onnx_graph.make_model("test", inputs.keys(), self.output_names)

        prepared_backend, ws = caffe2.python.onnx.backend.prepare(model_proto)

        if self.middle_input_names:
            for name, shape in self.middle_input_names.items():
                print("caffe2 middle input name:", name, "shape:", shape)
                break
            print("PREPROCESSOR_SUB:", PREPROCESSOR_SUB.shape)
            inputs_ssd = {name: PREPROCESSOR_SUB}
            result = prepared_backend.run(inputs_ssd)
        else:
            result = prepared_backend.run(inputs)

        if self.middle_output_names:
            self.show_result(result, "ssd_caffe2")
        else:
            self.show_result(result, "caffe2")

        if self.perf:
            start = time.time()
            for _ in range(PERFITER):
                _ = prepared_backend.run(inputs)
            self.onnx_runtime = time.time() - start
            print("self.onnx_runtime:", self.onnx_runtime)
        return result

    def run_caffe2_ssd(self, name, model_proto, inputs_ssd):
        """Run test again caffe2 backend."""
        print("----------------------run_caffe2()  ssd ---------------------------------------------")
        import caffe2.python.onnx.backend
        prepared_backend, ws = caffe2.python.onnx.backend.prepare(model_proto)

        result = prepared_backend.run(inputs_ssd)

        # TODO 后处理部分 ssd
        self.show_result(result, "ssd_caffe2")

        return result

    def run_onnxmsrt(self, name, onnx_graph, inputs):
        """Run test against onnxmsrt backend."""
        import lotus
        # create model and datafile in tmp path.
        model_path = os.path.join(TMPPATH, name + "_model.pb")
        model_proto = onnx_graph.make_model("test", inputs.keys(), self.output_names)
        with open(model_path, "wb") as f:
            f.write(model_proto.SerializeToString())
        m = lotus.ModelExecutor(model_path)
        results = m.run(self.output_names, inputs)
        if self.perf:
            start = time.time()
            for _ in range(PERFITER):
                _ = m.run(self.output_names, inputs)
            self.onnx_runtime = time.time() - start
        return results

    def run_onnxmsrtnext(self, name, onnx_graph, inputs):
        """Run test against msrt-next backend."""
        import lotus
        model_path = os.path.join(TMPPATH, name + ".pb")
        model_proto = onnx_graph.make_model("test", inputs.keys(), self.output_names)
        with open(model_path, "wb") as f:
            f.write(model_proto.SerializeToString())
        m = lotus.InferenceSession(model_path)
        results = m.run(self.output_names, inputs)
        if self.perf:
            start = time.time()
            for _ in range(PERFITER):
                _ = m.run(self.output_names, inputs)
            self.onnx_runtime = time.time() - start
        return results

    def run_onnxcntk(self, name, onnx_graph, inputs):
        """Run test against cntk backend."""
        import cntk as C
        model_path = os.path.join(TMPPATH, name + "_model.pb")
        model_proto = onnx_graph.make_model("test", inputs.keys(), self.output_names)
        with open(model_path, "wb") as f:
            f.write(model_proto.SerializeToString())
        z = C.Function.load(model_path, format=C.ModelFormat.ONNX)
        input_args = {}
        # FIXME: the model loads but eval() throws
        for arg in z.arguments:
            input_args[arg] = inputs[arg.name]
        results = z.eval(input_args)
        if self.perf:
            start = time.time()
            for _ in range(PERFITER):
                _ = z.eval(input_args)
            self.onnx_runtime = time.time() - start
        return results

    def create_onnx_file(self, name, onnx_graph, inputs, outdir):
        os.makedirs(outdir, exist_ok=True)
        model_path = os.path.join(outdir, name + ".onnx")
        # 如果配置文件中 出现 middle_output 的参数 ，则模型只转换到指定的 输出 node 输出 中间节点
        if self.middle_output_names:
            model_proto = onnx_graph.make_model(name, inputs.keys(), self.middle_output_names)
        else:
            model_proto = onnx_graph.make_model(name, inputs.keys(), self.output_names)

        with open(model_path, "wb") as f:
            f.write(model_proto.SerializeToString())
        print("\tcreated", model_path)

    def run_test(self, name, backend="caffe2", debug=False, onnx_file=None, opset=None, perf=None):
        """Run complete test against backend."""
        print("run_test name :", name)
        self.perf = perf

        # get the model
        if self.url:
            _, dir_name = self.download_file()
            model_path = os.path.join(dir_name, self.local)
        else:
            model_path = self.local
            dir_name = os.path.dirname(self.local)
        print("\tdownloaded", model_path)

        # if the input model is a checkpoint, convert it to a frozen model
        if self.model_type in ["checkpoint"]:
            saver = tf.train.import_meta_graph(model_path)
            with tf.Session() as sess:
                saver.restore(sess, model_path[:-5])
                frozen_graph = freeze_session(sess, output_names=self.output_names)
                tf.train.write_graph(frozen_graph, dir_name, "frozen.pb", as_text=False)
            model_path = os.path.join(dir_name, "frozen.pb")

        # create the input data
        inputs = self.make_input(self.input_names)

        print("self.output_names:", self.output_names)
        print("self.middle_output_names:", self.middle_output_names)
        print("self.input_names:", self.input_names)
        print("self.middle_input_names:", self.middle_input_names)

        # TODO creat middle inputs data like preprocessor/sub

        if self.more_inputs:
            for k, v in self.more_inputs.items():
                inputs[k] = v
        tf.reset_default_graph()
        graph_def = graph_pb2.GraphDef()
        print("model_path:", model_path)
        with open(model_path, "rb") as f:
            graph_def.ParseFromString(f.read())

        graph_def = tf2onnx.tfonnx.tf_optimize(None, inputs, self.output_names, graph_def)
        shape_override = {}
        g = tf.import_graph_def(graph_def, name='')

        with tf.Session(graph=g) as sess:
            # fix inputs if needed
            for k in inputs.keys():
                t = sess.graph.get_tensor_by_name(k)
                dtype = tf.as_dtype(t.dtype).name
                if type != "float32":
                    v = inputs[k]
                    inputs[k] = v.astype(dtype)
            if self.force_input_shape:
                shape_override = self.input_names

            # run the model with tensorflow
            # TODO optimize code
            if name == "ssd_mobile":
                tf_results = self.run_tensorflow_ssd(sess, inputs)
            else:
                tf_results = self.run_tensorflow(sess, inputs)

            onnx_graph = None
            print("\ttensorflow", "OK")
            try:
                #  重新构造ssd-onnx 所需要的 graph
                if name == "ssd_mobile":
                    pass
                else:
                    onnx_graph = self.to_onnx(sess.graph, opset=opset, shape_override=shape_override)
                    print("\t  onnx_graph", "OK")
                if debug:
                    onnx_graph.dump_graph()
                if onnx_file:
                    if name == "ssd_mobile":
                        pass
                    else:
                        self.create_onnx_file(name, onnx_graph, inputs, onnx_file)
                        print("\t  create_onnx_file", "OK")
            except Exception as ex:
                print("\tto_onnx", "FAIL", ex)


        # todo
        if self.middle_output_names:
            tf.reset_default_graph()
            graph_def1 = graph_pb2.GraphDef()
            with open(model_path, "rb") as f:
                graph_def1.ParseFromString(f.read())
            graph_def1 = tf2onnx.tfonnx.tf_optimize(None, inputs, self.middle_output_names, graph_def1)
            g1 = tf.import_graph_def(graph_def1, name='')

            with tf.Session(graph=g1) as sess1:

                print("---------------sess begin---------------------")
                try:
                    #  重新构造ssd-onnx 所需要的 graph
                    print("------ssd sess.graph------:", sess1.graph)
                    onnx_graph = self.to_onnx(sess1.graph, opset=opset, shape_override=shape_override)
                    print("onnx_graph:", onnx_graph)
                    print("\t  ------ssd onnx_graph------", "OK")
                    if debug:
                        onnx_graph.dump_graph()
                    if onnx_file:
                        self.create_onnx_file(name, onnx_graph, inputs, onnx_file)
                        print("\t ------ssd create_onnx_file------", "OK")
                except Exception as ex:
                    print("\t ------ssd to_onnx------", "FAIL", ex)


        try:
            onnx_results = None
            if backend == "caffe2":
                onnx_results = self.run_caffe2(name, onnx_graph, inputs)

                if name == "ssd_mobile":
                    #TODO 用 tensorflow 进行后处理

                    tf.reset_default_graph()
                    graph_def = graph_pb2.GraphDef()
                    print("model_path:", model_path)
                    with open(model_path, "rb") as f:
                        graph_def.ParseFromString(f.read())

                    graph_def = tf2onnx.tfonnx.tf_optimize(None, inputs, self.output_names, graph_def)
                    g = tf.import_graph_def(graph_def, name='')

                    self.run_tf_ssd_post(g, inputs)
                    print("run caffe2 post tf done")

            elif backend == "onnxmsrt":
                onnx_results = self.run_onnxmsrt(name, onnx_graph, inputs)
            elif backend == "onnxmsrtnext":
                onnx_results = self.run_onnxmsrtnext(name, onnx_graph, inputs)
            elif backend == "cntk":
                onnx_results = self.run_onnxcntk(name, onnx_graph, inputs)
            else:
                raise ValueError("unknown backend")
            print("\trun_onnx OK")

            try:
                if self.check_only_shape:
                    for i in range(len(tf_results)):
                        np.testing.assert_array_equal(tf_results[i].shape, onnx_results[i].shape)
                else:
                    for i in range(len(tf_results)):
                        np.testing.assert_allclose(tf_results[i], onnx_results[i], rtol=self.rtol, atol=self.atol)
                print("\tResults: OK")
                return True
            except Exception as ex:
                print("\tResults: ", ex)

        except Exception as ex:
            print("\trun_onnx", "FAIL", ex)

        return False


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="/tmp/pre-trained", help="pre-trained models cache dir")
    parser.add_argument("--config", default="tests/run_pretrained_models.yaml", help="yaml config to use")
    parser.add_argument("--tests", help="tests to run")
    parser.add_argument("--backend", default="caffe2",
                        choices=["caffe2", "onnxmsrt", "onnxmsrtnext", "cntk"], help="backend to use")
    parser.add_argument("--verbose", help="verbose output", action="store_true")
    parser.add_argument("--opset", type=int, default=None, help="opset to use")
    parser.add_argument("--debug", help="debug vlog", action="store_true")
    parser.add_argument("--list", help="list tests", action="store_true")
    parser.add_argument("--onnx-file", help="create onnx file in directory")
    parser.add_argument("--perf", help="capture performance numbers")
    parser.add_argument("--include-disabled", help="include disabled tests", action="store_true")
    args = parser.parse_args()
    return args


def tests_from_yaml(fname):
    tests = {}
    config = yaml.load(open(fname, 'r').read())
    for k, v in config.items():
        input_func = v.get("input_get")
        input_func = _INPUT_FUNC_MAPPING[input_func]
        kwargs = {}
        for kw in ["rtol", "atol", "disabled", "more_inputs", "check_only_shape", "model_type", "force_input_shape"]:
            if v.get(kw) is not None:
                kwargs[kw] = v[kw]

        test = Test(v.get("url"), v.get("model"), input_func, v.get("inputs"), v.get("outputs"), v.get("middle_inputs"), v.get("middle_outputs"), **kwargs)
        tests[k] = test
    return tests


def main():
    args = get_args()
    Test.cache_dir = args.cache
    tf2onnx.utils.ONNX_UNKNOWN_DIMENSION = 1
    tests = tests_from_yaml(args.config)
    if args.list:
        print(sorted(tests.keys()))
        return
    if args.tests:
        test_keys = args.tests.split(",")
    else:
        test_keys = list(tests.keys())

    failed = 0
    count = 0
    for test in test_keys:
        t = tests[test]
        if args.tests is None and t.disabled and not args.include_disabled:
            continue
        count += 1
        try:
            ret = t.run_test(test, backend=args.backend, debug=args.debug, onnx_file=args.onnx_file,
                             opset=args.opset, perf=args.perf)
        except Exception as ex:
            ret = None
            print(ex)
        if not ret:
            failed += 1

    print("=== RESULT: {} failed of {}, backend={}".format(failed, count, args.backend))

    if args.perf:
        with open(args.perf, "w") as f:
            f.write("test,tensorflow,onnx\n")
            for test in test_keys:
                t = tests[test]
                if t.perf:
                    f.write("{},{},{}\n".format(test, t.tf_runtime, t.onnx_runtime))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
