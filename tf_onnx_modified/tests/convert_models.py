# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import tarfile
import tempfile
import requests
import zipfile
import numpy as np
import tensorflow as tf
import tf2onnx
import yaml
from tensorflow.core.framework import graph_pb2
from tf2onnx.tfonnx import process_tf_graph
from tensorflow.python.framework.graph_util import convert_variables_to_constants

TMPPATH = tempfile.mkdtemp()
PERFITER = 1000

IMAGE_SHAPE = (0, 0)


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

    def __init__(self, url, local, input_names, output_names, middle_input_names=None, middle_output_names=None,
                 disabled=False, more_inputs=None, rtol=0.01, atol=0.,
                 check_only_shape=False, model_type="frozen", force_input_shape=False):
        self.url = url
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

    @staticmethod
    def to_onnx(tf_graph, opset=None, shape_override=None, input_names=None, middle_input_names=None):
        """Convert graph to tensorflow."""
        return process_tf_graph(tf_graph, continue_on_error=False, opset=opset, shape_override=shape_override,
                                inputs=input_names, middle_inputs=middle_input_names)

    def create_onnx_file(self, name, onnx_graph, inputs, outdir):
        print("--------------create_onnx_file-----begin-----")
        os.makedirs(outdir, exist_ok=True)
        model_path = os.path.join(outdir, name + ".onnx")
        # 如果配置文件中 出现 middle_output 的参数 ，则模型只转换到指定的 输出 node 输出 中间节点
        if self.middle_output_names:
            model_proto = onnx_graph.make_model(name, inputs, self.middle_output_names)
        else:
            model_proto = onnx_graph.make_model(name, inputs, self.output_names)

        with open(model_path, "wb") as f:
            f.write(model_proto.SerializeToString())
        print("\tcreated", model_path)

    def run_test(self, name, debug=False, onnx_file=None, opset=None, perf=None):
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

        input_names = self.input_names.keys()
        tf.reset_default_graph()
        graph_def = graph_pb2.GraphDef()
        print("model_path:", model_path)
        with open(model_path, "rb") as f:
            graph_def.ParseFromString(f.read())

        if self.middle_output_names:

            graph_def = tf2onnx.tfonnx.tf_optimize(None, input_names, self.middle_output_names, graph_def)
        else:
            graph_def = tf2onnx.tfonnx.tf_optimize(None, input_names, self.output_names, graph_def)

        with tf.Graph().as_default() as tf_graph:
            tf.import_graph_def(graph_def, name='')

        with tf.Session(graph=tf_graph) as sess:

            if self.force_input_shape:
                shape_override = self.middle_input_names
            try:
                #  重新构造ssd-onnx 所需要的 graph
                if self.middle_input_names:

                    middle_input_names = []
                    for name1, shape in self.middle_input_names.items():
                        middle_input_names.append(name1)
                    input_names = []
                    for name0, shape0 in self.input_names.items():
                        input_names.append(name0)

                    onnx_graph = self.to_onnx(sess.graph, opset=opset, shape_override=shape_override,
                                              input_names=input_names, middle_input_names=middle_input_names)
                    print("\t  ------ssd onnx_graph------", "OK")
                else:
                    onnx_graph = self.to_onnx(sess.graph, opset=opset)
                    print("\t  onnx_graph", "OK")
                if debug:
                    onnx_graph.dump_graph()
                if onnx_file:
                    self.create_onnx_file(name, onnx_graph, self.input_names, onnx_file)
                    print("\t  create_onnx_file", "OK")
            except Exception as ex:
                print("\tto_onnx", "FAIL", ex)

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
        kwargs = {}
        for kw in ["rtol", "atol", "disabled", "more_inputs", "check_only_shape", "model_type", "force_input_shape"]:
            if v.get(kw) is not None:
                kwargs[kw] = v[kw]

        test = Test(v.get("url"), v.get("model"), v.get("inputs"), v.get("outputs"), v.get("middle_inputs"),
                    v.get("middle_outputs"), **kwargs)
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
            ret = t.run_test(test, debug=args.debug, onnx_file=args.onnx_file,
                             opset=args.opset, perf=args.perf)
        except Exception as ex:
            ret = None
            print(ex)

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
