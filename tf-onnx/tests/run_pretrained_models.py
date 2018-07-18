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
import caffe2.python.onnx.backend
from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2

import cv2
from decode import decode
from utils import preprocess_image, postprocess, draw_detection, non_max_suppression
from config import anchors, class_names

TMPPATH = tempfile.mkdtemp()
PERFITER = 1000

IMAGE_SHAPE = (0, 0)



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
    "get_detection_raw": get_detection_raw
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

    def __init__(self, url, local, make_input, input_names, output_names,
                 disabled=False, more_inputs=None, rtol=0.01, atol=0.,
                 check_only_shape=False, model_type="frozen", force_input_shape=False):
        self.url = url
        self.make_input = make_input
        self.local = local
        self.input_names = input_names
        self.output_names = output_names
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
        print("run_tensorflow result:", result)
        self.show_result(result, "tensorflow")

        if self.perf:
            start = time.time()
            for _ in range(PERFITER):
                _ = sess.run(self.output_names, feed_dict=feed_dict)
            self.tf_runtime = time.time() - start
        return result

    @staticmethod
    def to_onnx(tf_graph, opset=None, shape_override=None):
        """Convert graph to tensorflow."""
        return process_tf_graph(tf_graph, continue_on_error=False, opset=opset, shape_override=shape_override)

    def run_caffe2(self, name, onnx_graph, inputs):
        """Run test again caffe2 backend."""
        print("run_caffe2()---------------------------------------------")
        import caffe2.python.onnx.backend
        model_proto = onnx_graph.make_model("test", inputs.keys(), self.output_names)
        prepared_backend = caffe2.python.onnx.backend.prepare(model_proto)
        result = prepared_backend.run(inputs)

        self.show_result(result, "caffe2")

        if self.perf:
            start = time.time()
            for _ in range(PERFITER):
                _ = prepared_backend.run(inputs)
            self.onnx_runtime = time.time() - start
            print("self.onnx_runtime:", self.onnx_runtime)
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
        model_proto = onnx_graph.make_model(name, inputs.keys(), self.output_names)
        with open(model_path, "wb") as f:
            f.write(model_proto.SerializeToString())
        print("\tcreated", model_path)

    def run_test(self, name, backend="caffe2", debug=False, onnx_file=None, opset=None, perf=None):
        """Run complete test against backend."""
        print(name)
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
        if self.more_inputs:
            for k, v in self.more_inputs.items():
                inputs[k] = v
        tf.reset_default_graph()
        graph_def = graph_pb2.GraphDef()
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
            tf_results = self.run_tensorflow(sess, inputs)

            onnx_graph = None
            print("\ttensorflow", "OK")
            try:
                # convert model to onnx
                onnx_graph = self.to_onnx(sess.graph, opset=opset, shape_override=shape_override)
                print("\tto_onnx", "OK")
                if debug:
                    onnx_graph.dump_graph()
                if onnx_file:
                    self.create_onnx_file(name, onnx_graph, inputs, onnx_file)
            except Exception as ex:
                print("\tto_onnx", "FAIL", ex)

        try:
            onnx_results = None
            if backend == "caffe2":
                onnx_results = self.run_caffe2(name, onnx_graph, inputs)
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

        test = Test(v.get("url"), v.get("model"), input_func, v.get("inputs"), v.get("outputs"), **kwargs)
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
    main()
