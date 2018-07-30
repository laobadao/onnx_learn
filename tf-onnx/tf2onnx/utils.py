# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.utils - misc utilities for tf2onnx
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from onnx import helper, onnx_pb
from tensorflow.core.framework import types_pb2, tensor_pb2

#
#  mapping dtypes from tensorflow to onnx
#
TF_TO_ONNX_DTYPE = {
    types_pb2.DT_FLOAT: onnx_pb.TensorProto.FLOAT,
    types_pb2.DT_HALF: onnx_pb.TensorProto.FLOAT16,
    types_pb2.DT_DOUBLE: onnx_pb.TensorProto.DOUBLE,
    types_pb2.DT_INT32: onnx_pb.TensorProto.INT32,
    types_pb2.DT_INT16: onnx_pb.TensorProto.INT16,
    types_pb2.DT_INT8: onnx_pb.TensorProto.INT8,
    types_pb2.DT_UINT8: onnx_pb.TensorProto.UINT8,
    types_pb2.DT_UINT16: onnx_pb.TensorProto.UINT16,
    types_pb2.DT_INT64: onnx_pb.TensorProto.INT64,
    types_pb2.DT_STRING: onnx_pb.TensorProto.STRING,
    types_pb2.DT_COMPLEX64: onnx_pb.TensorProto.COMPLEX64,
    types_pb2.DT_COMPLEX128: onnx_pb.TensorProto.COMPLEX128,
    types_pb2.DT_BOOL: onnx_pb.TensorProto.BOOL,
    types_pb2.DT_RESOURCE: onnx_pb.TensorProto.INT32,
}

#
# mapping dtypes from onnx to numpy
#
ONNX_TO_NUMPY_DTYPE = {
    onnx_pb.TensorProto.FLOAT: np.float32,
    onnx_pb.TensorProto.FLOAT16: np.float16,
    onnx_pb.TensorProto.DOUBLE: np.float64,
    onnx_pb.TensorProto.INT32: np.int32,
    onnx_pb.TensorProto.INT16: np.int16,
    onnx_pb.TensorProto.INT8: np.int8,
    onnx_pb.TensorProto.UINT8: np.uint8,
    onnx_pb.TensorProto.UINT16: np.uint16,
    onnx_pb.TensorProto.INT64: np.int64,
    onnx_pb.TensorProto.BOOL: np.bool,
}

#
#  onnx dtype names
#
ONNX_DTYPE_NAMES = {
    onnx_pb.TensorProto.FLOAT: "float",
    onnx_pb.TensorProto.FLOAT16: "float16",
    onnx_pb.TensorProto.DOUBLE: "double",
    onnx_pb.TensorProto.INT32: "int32",
    onnx_pb.TensorProto.INT16: "int16",
    onnx_pb.TensorProto.INT8: "int8",
    onnx_pb.TensorProto.UINT8: "uint8",
    onnx_pb.TensorProto.UINT16: "uint16",
    onnx_pb.TensorProto.INT64: "int64",
    onnx_pb.TensorProto.STRING: "string",
    onnx_pb.TensorProto.BOOL: "bool"
}

ONNX_UNKNOWN_DIMENSION = -1

#
# attributes onnx understands. Everything else coming from tensorflow
# will be ignored.
#
ONNX_VALID_ATTRIBUTES = {
    'p', 'bias', 'axes', 'pads', 'mean', 'activation_beta', 'spatial_scale', 'broadcast', 'pooled_shape', 'high',
    'activation_alpha', 'is_test', 'hidden_size', 'activations', 'beta', 'input_as_shape', 'drop_states', 'alpha',
    'momentum', 'scale', 'axis', 'dilations', 'transB', 'axis_w', 'blocksize', 'output_sequence', 'mode', 'perm',
    'min', 'seed', 'ends', 'paddings', 'to', 'gamma', 'width_scale', 'normalize_variance', 'group', 'ratio', 'values',
    'dtype', 'output_shape', 'spatial', 'split', 'input_forget', 'keepdims', 'transA', 'auto_pad', 'border', 'low',
    'linear_before_reset', 'height_scale', 'output_padding', 'shape', 'kernel_shape', 'epsilon', 'size', 'starts',
    'direction', 'max', 'clip', 'across_channels', 'value', 'strides', 'extra_shape', 'scales', 'k', 'sample_size',
    'blocksize', 'epsilon', 'momentum'
}

# index for internally generated names
INTERNAL_NAME = 1


def make_name(name):
    """Make op name for inserted ops."""
    global INTERNAL_NAME
    INTERNAL_NAME += 1
    return "{}__{}".format(name, INTERNAL_NAME)


def tf_to_onnx_tensor(tensor, name=""):
    """Convert tensorflow tensor to onnx tensor."""
    new_type = TF_TO_ONNX_DTYPE[tensor.dtype]
    tdim = tensor.tensor_shape.dim
    dims = [d.size for d in tdim]
    # FIXME: something is fishy here
    if dims == [0]:
        dims = [1]
    is_raw, data = get_tf_tensor_data(tensor)
    onnx_tensor = helper.make_tensor(name, new_type, dims, data, is_raw)
    return onnx_tensor


def get_tf_tensor_data(tensor):
    """Get data from tensor."""
    assert isinstance(tensor, tensor_pb2.TensorProto)
    is_raw = False
    if tensor.tensor_content:
        data = tensor.tensor_content
        is_raw = True
    elif tensor.float_val:
        data = tensor.float_val
    elif tensor.dcomplex_val:
        data = tensor.dcomplex_val
    elif tensor.int_val:
        data = tensor.int_val
    elif tensor.bool_val:
        data = tensor.bool_val
    elif tensor.dtype == tf.int32:
        data = [0]
    elif tensor.dtype == tf.int64:
        data = [0]
    elif tensor.dtype == tf.float32:
        data = [0.]
    elif tensor.string_val:
        data = tensor.string_val
    else:
        raise ValueError('tensor data not supported')
    return [is_raw, data]


def get_shape(node):
    """Get shape from tensorflow node."""
    # FIXME: do we use this?
    dims = None
    try:
        if node.type == "Const":
            shape = node.get_attr("value").tensor_shape
            dims = [int(d.size) for d in shape.dim]
        else:
            shape = node.get_attr("shape")
            dims = [d.size for d in shape.dim]
        if shape[0] is not None or shape[0] == -1:
            shape[0] = 1
    except Exception as ex:
        pass
    return dims


def map_tf_dtype(dtype):
    if dtype:
        dtype = TF_TO_ONNX_DTYPE[dtype]
    return dtype


def node_name(name):
    """Get node name without io#."""
    pos = name.find(":")
    if pos >= 0:
        return name[:pos]
    return name


def middle_node_shape(name):
    """ssd_mobile  shape  -1 or ? return run 后的 shape"""
    if name:
        shape = []
        if name in SSD_MIBILE_NODE_SHAPE.keys():
            shape = SSD_MIBILE_NODE_SHAPE[name]
        else:
            print("'", name, "':[],")
    return shape


SSD_MIBILE_NODE_SHAPE = {
    'image_tensor': [1, 300, 300, 3],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise': [1, 75, 75, 64],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNorm': [1, 75, 75, 64],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/Conv2D': [1, 150, 150, 32],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNorm': [1, 150, 150, 32],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/Relu6': [1, 150, 150, 32],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise': [1, 150, 150, 32],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNorm': [1, 150, 150, 32],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6': [1, 150, 150, 32],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2D': [1, 150, 150, 64],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNorm': [1, 150, 150, 64],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6': [1, 150, 150, 64],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/Relu6': [1, 75, 75, 64],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Relu6': [1, 75, 75, 128],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/Relu6': [1, 75, 75, 128],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6': [1, 75, 75, 128],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/Relu6': [1, 38, 38, 128],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu6': [1, 38, 38, 256],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/Relu6': [1, 38, 38, 256],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6': [1, 38, 38, 256],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/Relu6': [1, 19, 19, 256],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/Relu6': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/Relu6': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/Relu6': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/Relu6': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/Relu6': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/Relu6': [1, 10, 10, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Relu6': [1, 10, 10, 1024],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/Relu6': [1, 10, 10, 1024],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6': [1, 10, 10, 1024],
    'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_2_1x1_256/Relu6': [1, 10, 10, 256],
    'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512/Relu6': [1, 5, 5, 512],
    'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_3_1x1_128/Relu6': [1, 5, 5, 128],
    'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/Relu6': [1, 3, 3, 256],
    'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_4_1x1_128/Relu6': [1, 3, 3, 128],
    'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_4_3x3_s2_256/Relu6': [1, 2, 2, 256],
    'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_5_1x1_64/Relu6': [1, 2, 2, 64],
    'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_5_3x3_s2_128/Relu6': [1, 1, 1, 128],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2D': [1, 75, 75, 128],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNorm': [1, 75, 75, 128],

    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise': [1, 75, 75, 128],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNorm': [1, 75, 75, 128],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2D': [1, 75, 75, 128],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNorm': [1, 75, 75, 128],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise': [1, 38, 38, 128],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNorm': [1, 38, 38, 128],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2D': [1, 38, 38, 256],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNorm': [1, 38, 38, 256],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise': [1, 38, 38, 256],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNorm': [1, 38, 38, 256],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2D': [1, 38, 38, 256],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNorm': [1, 38, 38, 256],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise': [1, 19, 19, 256],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNorm': [1, 19, 19, 256],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2D': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNorm': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNorm': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2D': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNorm': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNorm': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2D': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNorm': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNorm': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2D': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNorm': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNorm': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2D': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNorm': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNorm': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2D': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNorm': [1, 19, 19, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise': [1, 10, 10, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNorm': [1, 10, 10, 512],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2D': [1, 10, 10, 1024],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNorm': [1, 10, 10, 1024],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise': [1, 10, 10, 1024],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNorm': [1, 10, 10, 1024],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2D': [1, 10, 10, 1024],
    'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNorm': [1, 10, 10, 1024],
    'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_2_1x1_256/Conv2D': [1, 10, 10, 256],
    'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_2_1x1_256/BatchNorm/FusedBatchNorm': [1, 10, 10, 256],
    'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512/Conv2D': [1, 5, 5, 512],
    'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512/BatchNorm/FusedBatchNorm': [1, 5, 5, 512],
    'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_3_1x1_128/Conv2D': [1, 5, 5, 128],
    'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_3_1x1_128/BatchNorm/FusedBatchNorm': [1, 5, 5, 128],
    'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/Conv2D': [1, 3, 3, 256],
    'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/BatchNorm/FusedBatchNorm': [1, 3, 3, 256],
    'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_4_1x1_128/Conv2D': [1, 3, 3, 128],
    'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_4_1x1_128/BatchNorm/FusedBatchNorm': [1, 3, 3, 128],
    'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_4_3x3_s2_256/Conv2D': [1, 2, 2, 256],
    'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_4_3x3_s2_256/BatchNorm/FusedBatchNorm': [1, 2, 2, 256],
    'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_5_1x1_64/Conv2D': [1, 2, 2, 64],
    'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_5_1x1_64/BatchNorm/FusedBatchNorm': [1, 2, 2, 64],
    'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_5_3x3_s2_128/Conv2D': [1, 1, 1, 128],
    'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_5_3x3_s2_128/BatchNorm/FusedBatchNorm': [1, 1, 1, 128],
    'BoxPredictor_5/BoxEncodingPredictor/Conv2D': [1, 1, 1, 24],
    'BoxPredictor_5/BoxEncodingPredictor/BiasAdd': [1, 1, 1, 24],
    'BoxPredictor_5/ClassPredictor/Conv2D': [1, 1, 1, 546],
    'BoxPredictor_5/ClassPredictor/BiasAdd': [1, 1, 1, 546],
    'BoxPredictor_4/BoxEncodingPredictor/Conv2D': [1, 2, 2, 24],
    'BoxPredictor_4/BoxEncodingPredictor/BiasAdd': [1, 2, 2, 24],
    'BoxPredictor_4/ClassPredictor/Conv2D': [1, 2, 2, 546],
    'BoxPredictor_4/ClassPredictor/BiasAdd': [1, 2, 2, 546],
    'BoxPredictor_3/BoxEncodingPredictor/Conv2D': [1, 3, 3, 24],
    'BoxPredictor_3/BoxEncodingPredictor/BiasAdd': [1, 3, 3, 24],
    'BoxPredictor_3/ClassPredictor/Conv2D': [1, 3, 3, 546],
    'BoxPredictor_3/ClassPredictor/BiasAdd': [1, 3, 3, 546],
    'BoxPredictor_2/BoxEncodingPredictor/Conv2D': [1, 5, 5, 24],
    'BoxPredictor_2/BoxEncodingPredictor/BiasAdd': [1, 5, 5, 24],
    'BoxPredictor_2/ClassPredictor/Conv2D': [1, 5, 5, 546],
    'BoxPredictor_2/ClassPredictor/BiasAdd': [1, 5, 5, 546],
    'BoxPredictor_1/BoxEncodingPredictor/Conv2D': [1, 10, 10, 24],
    'BoxPredictor_1/BoxEncodingPredictor/BiasAdd': [1, 10, 10, 24],
    'BoxPredictor_1/ClassPredictor/Conv2D': [1, 10, 10, 546],
    'BoxPredictor_1/ClassPredictor/BiasAdd': [1, 10, 10, 546],
    'BoxPredictor_0/BoxEncodingPredictor/Conv2D': [1, 19, 19, 12],
    'BoxPredictor_0/BoxEncodingPredictor/BiasAdd': [1, 19, 19, 12],
    'BoxPredictor_0/ClassPredictor/Conv2D': [1, 19, 19, 273],
    'BoxPredictor_0/ClassPredictor/BiasAdd': [1, 19, 19, 273],
    'BoxPredictor_0/Reshape': [1, 1083, 1, 4],
    'BoxPredictor_0/Reshape_1': [1, 1083, 91],
    'BoxPredictor_1/Reshape': [1, 600, 1, 4],
    'BoxPredictor_1/Reshape_1': [1, 600, 91],
    'BoxPredictor_2/Reshape': [1, 150, 1, 4],
    'BoxPredictor_2/Reshape_1': [1, 150, 91],
    'BoxPredictor_3/Reshape': [1, 54, 1, 4],
    'BoxPredictor_3/Reshape_1': [1, 54, 91],
    'BoxPredictor_4/Reshape': [1, 24, 1, 4],
    'BoxPredictor_4/Reshape_1': [1, 24, 91],
    'BoxPredictor_5/Reshape': [1, 6, 1, 4],
    'BoxPredictor_5/Reshape_1': [1, 6, 91],
    'concat': [1, 1917, 1, 4],
    'Squeeze': [1, 1917, 4],
    'concat_1': [1, 1917, 91],

    'Postprocessor/raw_box_encodings': [1, 1917, 4],
    'Postprocessor/Tile': [1, 1917, 4],
    'Postprocessor/Reshape': [1917, 4],
    'Postprocessor/Decode/get_center_coordinates_and_sizes/transpose': [4, 1917],
    'Postprocessor/Decode/get_center_coordinates_and_sizes/unstack': [1917],
    'Postprocessor/Decode/get_center_coordinates_and_sizes/sub': [1917],
    'Postprocessor/Decode/get_center_coordinates_and_sizes/sub_1': [1917],
    'Postprocessor/Reshape_1': [1917, 4],
    'Postprocessor/Decode/transpose': [4, 1917],
    'Postprocessor/Decode/unstack': [1917],
    'Postprocessor/Decode/get_center_coordinates_and_sizes/div': [1917],
    'Postprocessor/Decode/get_center_coordinates_and_sizes/add': [1917],
    'Postprocessor/Decode/get_center_coordinates_and_sizes/div_1': [1917],
    'Postprocessor/Decode/get_center_coordinates_and_sizes/add_1': [1917],
    'Postprocessor/Decode/div': [1917],
    'Postprocessor/Decode/mul_2': [1917],
    'Postprocessor/Decode/add': [1917],
    'Postprocessor/Decode/div_1': [1917],
    'Postprocessor/Decode/mul_3': [1917],
    'Postprocessor/Decode/add_1': [1917],
    'Postprocessor/Decode/div_2': [1917],
    'Postprocessor/Decode/Exp_1': [1917],
    'Postprocessor/Decode/mul_1': [1917],
    'Postprocessor/Decode/div_3': [1917],
    'Postprocessor/Decode/Exp': [1917],
    'Postprocessor/Decode/mul': [1917],
    'Postprocessor/Decode/div_4': [1917],
    'Postprocessor/Decode/sub': [1917],
    'Postprocessor/Decode/div_5': [1917],
    'Postprocessor/Decode/sub_1': [1917],
    'Postprocessor/Decode/div_6': [1917],
    'Postprocessor/Decode/add_2': [1917],
    'Postprocessor/Decode/div_7': [1917],
    'Postprocessor/Decode/add_3': [1917],
    'Postprocessor/Decode/stack': [4, 1917],
    'Postprocessor/Decode/transpose_1': [1917, 4],
    'Postprocessor/Reshape_2': [1, 1917, 4],
    'Postprocessor/raw_box_locations': [1, 1917, 4],
    'Postprocessor/ExpandDims_1': [1, 1917, 1, 4],
    'Postprocessor/scale_logits': [1, 1917, 91],
    'Postprocessor/convert_scores': [1, 1917, 91],
    'Postprocessor/raw_box_scores': [1, 1917, 91],
    'Postprocessor/Slice': [1, 1917, 90],

}
