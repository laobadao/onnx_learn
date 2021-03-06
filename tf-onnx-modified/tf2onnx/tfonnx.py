# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.tf2onnx - rewrite tensorflow graph to onnx graph
"""

from __future__ import division
from __future__ import print_function

import collections
import logging
import sys
import traceback

import numpy as np
import tf2onnx
from onnx import helper, onnx_pb, numpy_helper
from tensorflow.python.framework import graph_util
from tensorflow.tools.graph_transforms import TransformGraph
from tf2onnx import utils
from tf2onnx.graph import Node, Graph
from tf2onnx.graph_matcher import *
from tensorflow.core.framework import types_pb2

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tf2onnx")

TARGET_RS4 = "rs4"
TARGET_CAFFE2 = "caffe2"
DEFAULT_TARGET = [TARGET_RS4, TARGET_CAFFE2]


def delete_op_by_unsupport(ops):

    UN_SUPPORTED_OP_TYPE = ["Enter", "GreaterEqual", "LogicalAnd", "Assert", "Switch"]
    i = 0
    while i < len(ops):
        if ops[i].type in UN_SUPPORTED_OP_TYPE:
            ops.pop(i)
            i -= 1
            print("delete_op_by_unsupport len(ops):", len(ops))
        i += 1
    return ops

def delete_op_by_inputs_middle_inputs(ops, input, middle_input, nodes_outputs_dict):

    try:
        for node in ops:
            for i, tensor in enumerate(node.inputs):
                if tensor.name == input and node.name + ":0" != middle_input:
                    if node in ops:
                        ops.remove(node)
                    for out in node.outputs:
                        next_input = out.name
                        if node.name in nodes_outputs_dict.keys():
                            if nodes_outputs_dict[node.name] > 1:
                                for i in range(nodes_outputs_dict[node.name]-1):
                                    delete_op_by_inputs_middle_inputs(ops, next_input, middle_input, nodes_outputs_dict)

                        delete_op_by_inputs_middle_inputs(ops, next_input, middle_input, nodes_outputs_dict)

                if tensor.name == input and node.name + ":0" == middle_input:
                    if node in ops:
                        ops.remove(node)
    except Exception as ex:
        log.error("delete op from ops, %s op not in ops, ex: %s", node.name, ex)
        raise

    return ops


def tensorflow_to_onnx(graph, shape_override, inputs=None, middle_inputs=None):
    print("tensorflow_to_onnx")
    """
    Load tensorflow graph into an onnx graph with minimal rewrites so
    we can use the onnx graph as intermediate graph.
    """

    # ignore the following attributes
    ignored_attr = ["unknown_rank", "_class", "Tidx", "Tshape", "use_cudnn_on_gpu", "Index",
                    "Tpaddings", "TI", "Tparams", "Tindices", "Tlen", "Tdim", "dynamic_size", "element_shape",
                    "Tmultiples", "output_dtype", "Tblock_shape", "Tcrops", "index_type", "Taxis", "U",
                    "maxval", "Tout", "extrapolation_value", "method"]
    # some stats
    op_cnt = collections.Counter()
    attr_cnt = collections.Counter()
    onnx_nodes = []
    output_shapes = {}
    dtypes = {}

    # find outputs
    ops = graph.get_operations()
    new_input = None

    print("inputs:", inputs)
    print("middle_inputs:", middle_inputs)
    print("-----1----- raw  len(ops):", len(ops))

    inputs = inputs[0]

    if middle_inputs:
        nodes_outputs_dict = find_all_outputs(ops)
        ops = delete_op_by_inputs_middle_inputs(ops, inputs, middle_inputs[0], nodes_outputs_dict)
        ops = delete_op_by_unsupport(ops)

    print("-------2---------- final len(ops):", len(ops))
    delete_Identity_input = {}
    delete_Identity_output = {}

    # create dict with output to shape mappings
    for node in ops:

        if node.type == "Identity":
            identity_attrs = [a for a in node.node_def.attr]
            if "_class" not in identity_attrs:
                print("---node.inputs[0].name:", node.inputs[0].name)
                delete_Identity_input[node.name] = node.inputs[0].name
                print("---node.outputs[0].name:", node.outputs[0].name)
                delete_Identity_output[node.name] = node.outputs[0].name

        for out in node.outputs:
            shape = shape_override.get(out.name)
            if shape is None:
                try:
                    shape = out.get_shape().as_list()
                except Exception as ex:
                    shape = []

            if shape:
                for i, v in enumerate(shape):
                    if v is None:
                        shape[i] = -1
                if shape[0] == -1:
                    shape[0] = utils.ONNX_UNKNOWN_DIMENSION

            if middle_inputs:
                new_input = middle_inputs[0]

            if middle_inputs and node.type == "Placeholder":
                dtypes[new_input] = utils.map_tf_dtype(types_pb2.DT_FLOAT)
                if shape_override:
                    shape = shape_override.get(new_input)
                    if shape is None:
                        shape = shape_override.get(node.name + ":0")

                output_shapes[new_input] = shape
                output_shapes[out.name] = shape
            elif node.type == "CropAndResize":
                dtypes[new_input] = utils.map_tf_dtype(types_pb2.DT_FLOAT)
                output_shapes[new_input] = shape
                output_shapes[out.name] = shape
            else:
                dtypes[out.name] = utils.map_tf_dtype(out.dtype)
                output_shapes[out.name] = shape

    print("delete_Identity_output:", delete_Identity_output)
    print("delete_Identity_input:", delete_Identity_input)
    # minimal conversion of attributes
    for node in ops:
        attr = {}
        takeit = True
        op_cnt[node.type] += 1
        for a in node.node_def.attr:
            attr_cnt[a] += 1
            if a == "dtype":
                attr[a] = utils.map_tf_dtype(node.get_attr("dtype"))
            elif a == "T":
                dtype = node.get_attr("T")
                if dtype:
                    if not isinstance(dtype, list):
                        dtypes[node.name] = utils.map_tf_dtype(dtype)
            elif a in ["output_type", "output_dtype", "out_type"]:
                attr[a] = utils.map_tf_dtype(node.get_attr(a))
            elif a == "shape":
                attr[a] = utils.get_shape(node)
            elif a == "Tperm":
                pass
            elif a == "_output_shapes":
                attr[a] = utils.get_shape(node)
            elif a == "value":
                onnx_tensor = utils.tf_to_onnx_tensor(node.get_attr(a), name=node.name + ":0")
                attr[a] = onnx_tensor
            elif a == "DstT":
                attr["to"] = utils.map_tf_dtype(node.get_attr("DstT"))
            elif a == "SrcT":
                continue
            elif a in ignored_attr:
                continue
            else:
                attr[a] = node.get_attr(a)

        if takeit:
            try:
                input_names = [i.name for i in node.inputs]
                output_names = [i.name for i in node.outputs]

                if middle_inputs and node.type == "Placeholder":
                    print("attr before:", attr)
                    attr["dtype"] = utils.map_tf_dtype(types_pb2.DT_FLOAT)
                    # output_shapes[new_input]
                    if shape_override:
                        shape = shape_override.get(new_input)
                        if shape is None:
                            shape = shape_override.get(node.name + ":0")
                        attr["shape"] = shape
                        print("attr[shape]", shape)
                    else:
                        attr["shape"] = output_shapes[new_input]
                    output_names = [new_input]
                    print("output_names:", output_names)
                    new_name = new_input[:-2]
                    print("new_name:", new_name)
                    print("attr after:", attr)
                    onnx_node = helper.make_node(node.type, input_names, output_names, name=new_name, **attr)
                elif middle_inputs and node.type == "CropAndResize":
                    attr["dtype"] = utils.map_tf_dtype(types_pb2.DT_FLOAT)
                    attr["shape"] = output_shapes[new_input]
                    print("attr before:", attr)
                    output_names = [new_input]
                    print("output_names:", output_names)
                    new_name = new_input[:-2]
                    print("new_name:", new_name)
                    print("attr after:", attr)
                    new_type = "Placeholder"
                    onnx_node = helper.make_node(new_type, input_names, output_names, name=new_name, **attr)
                else:
                    # 找到 Identity 后面的 op 将 Identity 的输入赋值给后面 op 的输入，直接忽略 Identity
                    if len(input_names) >= 2 and delete_Identity_output is not None:
                        for k, v in delete_Identity_output.items():
                            if input_names[0] == v:
                                input_names[0] = delete_Identity_input[k]
                                print("-----find it---", input_names)

                    onnx_node = helper.make_node(node.type, input_names, output_names, name=node.name, **attr)

                onnx_nodes.append(onnx_node)
            except Exception as ex:
                log.error("pass 1 convert failed for %s, ex=%s", node, ex)
                raise

    return onnx_nodes, op_cnt, attr_cnt, output_shapes, dtypes


def find_all_outputs(ops):

    nodes_outputs_dict = {}

    for node in ops:
        if node.inputs:
            node_outputs = []
            for node_1 in ops:
                for tensor in node_1.inputs:
                    if tensor.name == node.name + ":0" or tensor.name == node.name + ":1":
                        node_outputs.append(node_1)
            nodes_outputs_dict[node.name] = len(node_outputs)

    return nodes_outputs_dict


def _convert_shapenode_to_int64(ctx, node, input_number):
    shape_node = node.inputs[1]
    name = node.input[1]
    if shape_node.is_const():
        # if it is a const, change the const to be int64
        shape = shape_node.get_tensor_value()
        shape = np.array(list(shape), dtype=np.int64)
        onnx_tensor = numpy_helper.from_array(shape, name)
        ctx._initializers[name] = onnx_tensor
        shape_node.set_attr("value", onnx_tensor)
        return [node]
    else:
        cast_op = ctx.insert_new_node_on_input(node, "Cast", name)
        cast_op.set_attr("to", onnx_pb.TensorProto.INT64)
        ctx.copy_shape(name, cast_op.output[0])
        return [cast_op, node]


# pylint: disable=W0613,C0111,W0612


def no_op(ctx, node, name, args):
    """Skip node."""
    return None


def direct_op(ctx, node, name, args):
    """Take node as is, no updates required"""
    return node


def leaky_relu_op(ctx, node, name, args):
    """Take node as is, no updates required"""
    # print("node:", node)
    return node


def dense_op(ctx, node, name, args):
    """Take node as is, no updates required"""
    # print("node:", node)
    node.type = "Dense"
    print("------------dense_op-----------------")
    return node


def identity_op(ctx, node, name, args):
    """Identity."""
    if node.inputs[0]:
        if node.inputs[0].is_const():
            # if identity has a const as input, remove it
            input_name = node.input[0]
            output_name = node.output[0]
            for n in ctx.get_nodes():
                for i, parent_name in enumerate(n.input):
                    if parent_name == output_name:
                        n.input[i] = input_name
            return None
    print("------------------Identity:", node)

    return None


def broadcast_op(ctx, node, name, args):
    """Elementwise Ops with broadcast flag."""
    shape0 = ctx.get_shape(node.input[0])
    shape1 = ctx.get_shape(node.input[1])
    if shape0 != shape1:
        node.set_attr("broadcast", 1)
        # this is more shortcoming in the broadcasting code
        # of caffe2 and winml/rs4 but since those are the primary
        # onnx-1.1 backends we don't use a seperate flag.
        if shape0 is not None and len(shape0) == 0:
            if node.inputs[0].is_const():
                shape0 = node.inputs[0].scalar_to_dim1()
        if shape1 is not None and len(shape1) == 0:
            if node.inputs[1].is_const():
                shape1 = node.inputs[1].scalar_to_dim1()
        if shape0 and shape1 and len(shape0) < len(shape1) and node.type in ["Mul", "Add"]:
            tmp = node.input[0]
            node.input[0] = node.input[1]
            node.input[1] = tmp
    else:
        node.set_attr("broadcast", 0)
    return node


def broadcast_op7(ctx, node, name, args):
    """Elementwise Ops with broadcast flag."""
    shape0 = ctx.get_shape(node.input[0])
    shape1 = ctx.get_shape(node.input[1])
    if shape0 != shape1:
        # this is more shortcoming in the broadcasting code
        # of caffe2 and winml/rs4 but since those are the primary
        # onnx-1.1 backends we don't use a seperate flag.
        if shape0 is not None and len(shape0) == 0:
            if node.inputs[0].is_const():
                shape0 = node.inputs[0].scalar_to_dim1()
        if shape1 is not None and len(shape1) == 0:
            if node.inputs[1].is_const():
                shape1 = node.inputs[1].scalar_to_dim1()
        if shape0 and shape1 and len(shape0) < len(shape1) and node.type in ["Mul", "Add"]:
            tmp = node.input[0]
            node.input[0] = node.input[1]
            node.input[1] = tmp
        if shape0 and shape1 and len(shape0) == len(shape1) and node.type in ["Mul", "Add"]:
            if shape0[len(shape0) - 1] < shape1[len(shape1) - 1]:
                tmp = node.input[0]
                node.input[0] = node.input[1]
                node.input[1] = tmp

    return node


def const_op(ctx, node, name, args):
    """Constants - make those initializers. 权重"""
    tensor = node.get_attr("value")
    ctx.add_initializer(tensor.t)
    # we return None - const will not be in the node list. But we keep the mapping for
    # get_node_by_name() so we don't need to lookup the initializers.
    return None


def arg_minmax_op(ctx, node, name, args):
    # output_type output = ArgMin(T input, Tidx dimension, @type Tidx, @type output_type)
    # tensor(int32) reduced = ArgMin(T data, @INT axis, @INT keepdims)
    axis_node = node.inputs[1]
    axis = axis_node.get_tensor_value()
    node.set_attr("axis", axis[0])
    ctx.remove_input(node, node.input[1])
    return node


def reduce_op(ctx, node, name, args):
    axes_node = node.inputs[1]
    axis = axes_node.get_tensor_value()
    node.set_attr("axes", axis)
    ctx.remove_input(node, node.input[1])
    keep_dims = node.get_attr("keep_dims")
    if keep_dims:
        del node.attr['keep_dims']
        node.set_attr("keepdims", keep_dims.i)
    return node


def placeholder_op(ctx, node, name, args):
    print("------placeholder_op-----method----")
    print("placeholder_op:", node)
    print("node.output[0]:", node.output[0])
    print("node.dtype:", node.dtype)
    print("node.shape:", node.shape)
    input_node = helper.make_tensor_value_info(node.output[0], node.dtype, ctx.get_shape(node.output[0]))
    ctx.model_inputs.append(input_node)
    return None


def square_op(ctx, node, name, args):
    node.type = "Mul"
    node.input.append(node.input[0])
    return node


def squeeze_op(ctx, node, name, args):
    # T output = Squeeze(T input, @list(int) squeeze_dims)
    # T squeezed = Squeeze(T data, @AttrType.INTS axes)
    axis = node.get_attr("axis")
    if not axis:
        axis = node.get_attr("squeeze_dims")
        if axis:
            del node.attr["squeeze_dims"]
    else:
        del node.attr["axis"]

    shape = ctx.get_shape(node.input[0])
    if axis and axis.ints:
        axis = axis.ints
    else:
        axis = [i for i, j in enumerate(shape) if j == 1]
    node.set_attr("axes", axis)
    return node


NCHW_TO_NHWC = [0, 2, 3, 1]
NHWC_TO_NCHW = [0, 3, 1, 2]
HWCN_TO_NCHW = [3, 2, 0, 1]
NCHW_TO_HWCN = [2, 3, 1, 0]


def spatial_map(shape, perm):
    new_shape = shape[:]
    for i in perm:
        new_shape[i] = shape[perm[i]]
    return new_shape


def add_padding(ctx, node):

    padding = node.get_attr("padding")
    if padding:
        padding = padding.s.decode("utf-8")
        node.set_attr("pads", padding)


def conv_dims_attr(node, name, new_name=None):
    if new_name is None:
        new_name = name
    dims = node.get_attr(name)
    if not dims:
        return None
    dims = dims.ints
    if node.is_nhwc():
        if len(dims) == 2:
            h, w = dims
            c = n = 1
        else:
            n, h, w, c = dims
    else:
        n, c, h, w = dims
    dims = [h, w]
    node.set_attr(new_name, dims)
    return dims


def conv_kernel_shape(ctx, node, input_idx, spatial=2):
    kernel_shape = ctx.get_shape(node.input[input_idx])
    if len(kernel_shape) != 2 * spatial:
        raise ValueError("kernel rank must be 2* spatial")
    kernel_shape = kernel_shape[0:spatial]
    node.set_attr("kernel_shape", kernel_shape)
    return kernel_shape


def conv_op(ctx, node, name, args):
    # print("conv_op")
    # T output = Conv2D(T input, T filter, @list(int) strides, @bool use_cudnn_on_gpu,
    #                       @string padding, @string data_format)
    # T Y = Conv(T X, T W, T B, @AttrType.STRING auto_pad, @AttrType.INTS dilations, @AttrType.INT group,
    #                       @AttrType.INTS kernel_shape, @AttrType.INTS pads, @AttrType.INTS strides)
    kernel_shape = conv_kernel_shape(ctx, node, 1, spatial=2)
    strides = conv_dims_attr(node, "strides")
    dilations = conv_dims_attr(node, "dilations")

    add_padding(ctx, node)
    return node


def convtranspose_op(ctx, node, name, args):
    # T output = Conv2DBackpropInput(int32 input_sizes, T filter, T out_backprop,
    #    @list(int) strides, @bool use_cudnn_on_gpu, @string padding, @string data_format, @list(int) dilations)
    # T Y = ConvTranspose(T X, T W, T B, @STRING auto_pad, @INTS dilations,
    #    @INT group, @INTS kernel_shape, @INTS output_shape, @INTS pads, @INTS strides)

    # Note: inputs are reversed from what one would expect.
    kernel_shape = conv_kernel_shape(ctx, node, 1)

    # ouput_shape is explicitly specified here, in this case pads values are auto generated/calculated.
    output_shape = node.inputs[0].get_tensor_value()
    if node.is_nhwc():
        new_output_shape = [output_shape[1], output_shape[2]]
    else:
        new_output_shape = [output_shape[2], output_shape[3]]
    node.set_attr("output_shape", new_output_shape)

    strides = conv_dims_attr(node, "strides")
    conv_dims_attr(node, "dilations")

    # remove output_shapes input
    ctx.remove_input(node, node.input[0])
    # swap data and kernel
    t = node.input[0]
    node.input[0] = node.input[1]
    node.input[1] = t

    # nodes = conv_convert_inputs(ctx, node, with_kernel=True)

    # Note: output_padding, group are left default.
    return node


def depthwiseconv_op(ctx, node, name, args):
    # T output = DepthwiseConv2dNative(T input, T filter, @list(int) strides, @string padding, @string data_format)
    # T Y = ConvTranspose(T X, T W, T B, @AttrType.STRING auto_pad, @AttrType.INTS dilations, @AttrType.INT group,
    #           @AttrType.INTS kernel_shape, @AttrType.INTS output_shape, @AttrType.INTS pads, @AttrType.INTS strides)
    #
    # this is not documented well in onnx, the hint comes from pytorch documentation:
    # http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
    #   The configuration when groups == in_channels and out_channels = K * in_channels
    #   where K is a positive integer is termed in literature as depthwise convolution.
    #   In other words, for an input of size (N,Cin,Hin,Win),
    #   if you want a depthwise convolution with a depthwise multiplier K,
    #   then you use the constructor arguments (in_channels=Cin,out_channels=Cin*K,...,groups=Cin)
    #
    node.type = "DepthwiseConv2d"
    input_shape = ctx.get_shape(node.input[0])
    if len(input_shape) != 4:
        raise ValueError("only Conv2D is supported")

    if node.is_nhwc():
        i_n, i_h, i_w, i_c = input_shape
    else:
        i_n, i_c, i_h, i_w = input_shape

    kernel_shape = ctx.get_shape(node.input[1])
    if len(kernel_shape) != 4:
        raise ValueError("only Conv2D is supported")

    k_h, k_w, k_input_channels, k_channel_multiplier = kernel_shape

    node.set_attr("kernel_shape", [k_h, k_w])
    strides = conv_dims_attr(node, "strides")
    conv_dims_attr(node, "dilations")
    node.set_attr("group", i_c)
    add_padding(ctx, node)
    return node


def depthwiseconv2dnative_op(ctx, node, name, args):
    node.type = "DepthwiseConv2dNative"
    input_shape = ctx.get_shape(node.input[0])
    if len(input_shape) != 4:
        raise ValueError("only Conv2D is supported")

    if node.is_nhwc():
        i_n, i_h, i_w, i_c = input_shape
    else:
        i_n, i_c, i_h, i_w = input_shape

    kernel_shape = ctx.get_shape(node.input[1])
    if len(kernel_shape) != 4:
        raise ValueError("only Conv2D is supported")

    k_h, k_w, k_input_channels, k_channel_multiplier = kernel_shape

    node.set_attr("kernel_shape", [k_h, k_w])
    strides = conv_dims_attr(node, "strides")
    conv_dims_attr(node, "dilations")
    node.set_attr("group", i_c)
    add_padding(ctx, node)

    return node


def pool_op(ctx, node, name, args):
    # T output = MaxPool(T input, @list(int) ksize, @list(int) strides, @string padding, @string data_format)
    # T Y = MaxPool(T X, @AttrType.STRING auto_pad, @AttrType.INTS kernel_shape, @AttrType.INTS pads,
    #                   @AttrType.INTS strides)
    # above seems wrong - input[1] is ksize, input[2] is strides
    if len(node.input) < 3:
        kernel_shape = node.get_attr("ksize").ints
        kernel_shape = [kernel_shape[1], kernel_shape[2]]
        node.set_attr("kernel_shape", kernel_shape)
        strides = conv_dims_attr(node, "strides")
    else:
        kernel_shape = node.inputs[1].get_tensor_value()
        kernel_shape = [kernel_shape[1], kernel_shape[2]]
        node.set_attr("kernel_shape", kernel_shape)

        strides = node.inputs[2].get_tensor_value()
        strides = [strides[1], strides[2]]
        node.set_attr("strides", strides)

        ctx.remove_input(node, node.input[2])
        ctx.remove_input(node, node.input[1])

    conv_dims_attr(node, "dilations")
    add_padding(ctx, node)

    return node


def new_relu6_op(ctx, node, name, args):
    node.type = "Relu6"
    return node


def squareddifference_op(ctx, node, name, args):
    node.type = "Sub"
    op_name = utils.make_name(node.name)
    mul = ctx.insert_new_node_on_output("Mul", node.output[0], name=op_name)
    mul.input.append(node.output[0])
    return [node, mul]


def cast_op(ctx, node, name, args):
    # DstT y = Cast(SrcT x, @type SrcT, @type DstT)
    # T2 output = Cast(T1 input, @STRING to)
    dst = node.get_attr("to")
    dst = tf2onnx.utils.ONNX_DTYPE_NAMES[dst]
    node.set_attr("to", dst)
    return node


def biasadd_op(ctx, node, name, args):
    # T output = BiasAdd(T value, T bias, @string data_format)
    # T output = BiasAddV1(T value, T bias)
    #  这里 为了转 Keras 要 把 name 换成 AddBias
    node.type = "AddBias"
    return broadcast_op(ctx, node, name, args)


def biasadd_op7(ctx, node, name, args):
    # T output = BiasAdd(T value, T bias, @string data_format)
    # T output = BiasAddV1(T value, T bias)
    #  这里 为了转 Keras 要 把 name 换成 AddBias
    node.type = "AddBias"
    return broadcast_op7(ctx, node, name, args)


def transpose_op(ctx, node, name, args):
    # T y = Transpose(T x, Tperm perm, @type Tperm)
    # T transposed = Transpose(T data, @INTS perm)
    if len(node.input) > 1:
        perm = node.inputs[1]
        if perm.is_const():
            # perms is passed as const
            dims = perm.get_tensor_value()
        else:
            # calculate perms from shape
            shape = ctx.get_shape(node.input[1])
            dims = [i for i in range(len(shape) - 1, -1)]
        ctx.remove_input(node, node.input[1])
        node.set_attr("perm", dims)
    else:
        # graph rewrite moved perm to attribute
        pass
    return node


def concat_op(ctx, node, name, args):
    # T output = ConcatV2(T values, Tidx axis, @int N, @type Tidx)
    # T concat_result = Concat(T inputs, @INT axis)
    axis_node = node.inputs[-1]
    axis = axis_node.get_tensor_value()
    ctx.remove_input(node, node.input[-1])
    node.set_attr("axis", axis[0])
    return node


def gatherv2_op(ctx, node, name, args):
    # for GatherV2 axis come as input
    axis = node.inputs[2].get_tensor_value()
    ctx.remove_input(node, node.input[2])
    node.set_attr("axis", axis[0])
    return node


def split_op(ctx, node, name, args):
    # T output = Split(int32 split_dim, T value, @int num_split)
    # T outputs = Split(T input, @INT axis, @INTS split)
    split_dims = node.inputs[0].get_tensor_value()
    ctx.remove_input(node, node.input[0])
    node.set_attr("axis", split_dims[0])
    return node


def splitv_op(ctx, node, name, args):
    # T output = SplitV(T value, Tlen size_splits, int32 split_dim, @int num_split, @type Tlen)
    # T outputs = Split(T input, @INT axis, @INTS split)
    shape = ctx.get_shape(node.input[0])
    split = node.inputs[1].get_tensor_value()
    split_dims = node.inputs[2].get_tensor_value()
    ctx.remove_input(node, node.input[2])
    ctx.remove_input(node, node.input[1])
    # onnx-caffe2 caffe2 格式 split 都是正数 [1,1,1,1-1]-->[1,1,1,1,81]  相加为85
    if len(split) == 5 and split[-1] == -1:
        split = np.array(
            [split[0], split[1], split[2], split[3], shape[-1] - (split[0] + split[1] + split[2] + split[3])])

    node.set_attr("split", split)
    node.set_attr("axis", split_dims[0])
    return node


def pad_op(ctx, node, name, args):
    """
    Tensorflow 中 Pad op mode：取值为 "CONSTANT"、"REFLECT" 或 "SYMMETRIC"（不区分大小写）,
    我们仅支持 CONSTANT mode 且 value 为 0 的情况。

    :param ctx:
    :param node:
    :param name:
    :param args:
    :return:
    """
    # T output = Pad(T input, int32 paddings, @type Tpaddings), CONST model using default value
    #  or PadV2(T input, int32 paddings, T constant_value, @type Tpaddings), CONST mode - default value specified
    #  or MirrorPad(T input, int32 paddings, @type Tpaddings, @STRING mode), other mode.
    # T output = Pad(T data, @STRING mode, @INTS pads, @FLOAT value)
    # print("----------------------------pad_op------------------------------:\n", node)
    paddings = np.array(node.inputs[1].get_tensor_value()).transpose().flatten()
    mode = node.get_attr("mode")
    if mode:
        mode = mode.s.decode("utf-8").lower()
        node.set_attr("mode", mode)
    if mode not in [None, "constant"]:
        raise ValueError(mode + " pad mode is not supported")

    if mode in [None, "constant"] and len(node.input) == 3:
        const_val = node.inputs[2].get_tensor_value()[0]
        print("const_val")
        node.set_attr("value", const_val)
        ctx.remove_input(node, node.input[2])

    ctx.remove_input(node, node.input[1])
    node.set_attr("pads", paddings)
    return node


def rsqrt_op(ctx, node, name, args):
    print("------------rsqrt_op------------")
    node.type = "Sqrt"
    op_name = utils.make_name(node.name)
    reciprocal = ctx.insert_new_node_on_output("Reciprocal", node.output[0], name=op_name)
    ctx.copy_shape(node.output[0], reciprocal.output[0])
    return [node, reciprocal]


def expanddims_op(ctx, node, name, args):
    # T output = ExpandDims(T input, Tdim dim, @type Tdim)
    # tensorflow already inferes the output shape so we can just take it
    shape = ctx.get_shape(node.output[0])
    node.type = "Reshape"
    ctx.remove_input(node, node.input[1])
    node.set_attr("shape", shape)
    return node


def expanddims_op7(ctx, node, name, args):
    shape = ctx.get_shape(node.output[0])
    shape_name = utils.make_name(node.name)
    shape_node = ctx.make_const(shape_name, "Const", np.array(shape, dtype=np.int64))
    node.type = "Reshape"
    node.input[1] = shape_name
    return node


def slice_op(ctx, node, name, args):
    # T output = Slice(T input, Index begin, Index size, @type Index)
    # T output = Slice(T data, @INTS axes, @INTS ends, @INTS starts)
    starts = node.inputs[1].get_tensor_value()
    size = node.inputs[2].get_tensor_value()
    shape = ctx.get_shape(node.input[0])
    # 原作者 这部分没有考虑 -1 的情况 op 的属性不同 需要根据 shape 计算 for caffe2 run ,修改 size 重新计算 ends ，tensorflow 自己会算
    new_size = []
    for i in range(len(size)):
        if size[i] == -1:
            new_size.append(shape[i] - starts[i])
        else:
            new_size.append(size[i])
    ends = np.add(starts, new_size)
    ctx.remove_input(node, node.input[2])
    ctx.remove_input(node, node.input[1])
    node.set_attr("starts", starts)
    node.set_attr("ends", ends)
    return node


def stridedslice_op(ctx, node, name, args):
    # for now we implement common cases. Things like strides!=1 are not mappable to onnx.
    node.type = "Slice"
    not_supported_attr = ["ellipsis_mask", "new_axis_mask"]
    for attr_name in not_supported_attr:
        attr = node.get_attr(attr_name)
        if attr is not None and attr.i != 0:
            raise ValueError("StridedSlice: attribute " + attr_name + " not supported")

    begin = node.inputs[1].get_tensor_value()
    end = node.inputs[2].get_tensor_value()
    strides = node.inputs[3].get_tensor_value()
    end_mask = node.get_attr("end_mask")
    end_mask = end_mask.i if end_mask is not None else 0
    shrink_axis_mask = node.get_attr("shrink_axis_mask")
    shrink_axis_mask = shrink_axis_mask.i if shrink_axis_mask is not None else 0
    new_begin = []
    new_end = []
    axes = []
    # onnx slice op can't remove a axis, track axis and add a squeeze op if needed
    needs_squeeze = []

    for idx in range(len(begin)):
        if strides[idx] != 1:
            raise ValueError("StridedSlice: only strides=1 is supported")
        axes.append(idx)
        mask = (shrink_axis_mask >> idx) & 1
        if mask != 0:
            new_begin.append(begin[idx])
            new_end.append(end[idx])
            needs_squeeze.append(idx)
            continue

        new_begin.append(begin[idx])
        mask = (end_mask >> idx) & 1
        if mask != 0:
            new_end.append(sys.maxsize)
        else:
            new_end.append(end[idx])

    node.set_attr("starts", begin)
    node.set_attr("ends", end)
    node.set_attr("axes", axes)
    ctx.remove_input(node, node.input[3])
    ctx.remove_input(node, node.input[2])
    ctx.remove_input(node, node.input[1])

    return node



def lrn_op(ctx, node, name, args):
    # FIXME: numerical results are not correct
    # ONNX: Each input value is divided by (bias+(alpha/size)*sum(xi^2 for every xi in the local region))^beta
    # TF: sqr_sum[a, b, c, d] = sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
    #     output = input / (bias + alpha * sqr_sum) ** beta
    depth_radius = node.get_attr("depth_radius")
    if depth_radius:
        size = depth_radius.i
    else:
        size = 5
    node.set_attr("size", size)
    return node


def upsample_op7(ctx, node, name, args):
    # "UpSampling2D"
    node.type = "UpSampling2D"
    shape = ctx.get_shape(node.input[0])
    target_shape = node.inputs[1].get_tensor_value()
    n, h, w, c = shape
    nh, nw = target_shape
    size = [int(float(nh) / h), int(float(nw) / w)]
    node.set_attr("size", size)
    ctx.remove_input(node, node.input[1])
    return node


def multinomial_op(ctx, node, name, args):
    # output_dtype output = Multinomial(T logits, int32 num_samples, @int seed, @int seed2, @type output_dtype)
    sample_size = node.inputs[1].get_tensor_value()
    seed = node.get_attr("seed")
    if seed:
        node.set_attr("seed", float(seed.i))
    output_dtype = node.get_attr("output_dtype")
    if output_dtype:
        output_dtype = output_dtype.i
    else:
        output_dtype = onnx_pb.TensorProto.INT32
    node.set_attr("dtype", output_dtype)
    node.set_attr("sample_size", sample_size[0])
    ctx.remove_input(node, node.input[1])
    return node


def topk_op(ctx, node, name, args):
    k = node.inputs[1].get_tensor_value()
    node.set_attr("k", k[0])
    node.type = "TopK"
    ctx.remove_input(node, node.input[1])

    # the second of TopK operator must be INT64 per ONNX requires.
    ctx.set_dtype(name + ":1", onnx_pb.TensorProto.INT64)
    return node


def tile_op7(ctx, node, name, args):
    # onnx wants shape input to be int64
    return _convert_shapenode_to_int64(ctx, node, 1)


def spacetodepth_op(ctx, node, name, args):
    block_size = node.get_attr("block_size")
    node.set_attr("blocksize", block_size.i)
    # nodes = conv_convert_inputs(ctx, node, with_kernel=False)
    return node


def reshape_op5(ctx, node, name, args):
    need_casting = node.dtype in [onnx_pb.TensorProto.INT32,
                                  onnx_pb.TensorProto.INT16,
                                  onnx_pb.TensorProto.INT64]

    # onnx wants reshape.input[1] to have the value be int64 which is not the case for tensorflow.
    nodes = _convert_shapenode_to_int64(ctx, node, 1)
    if not need_casting:
        # onnx reshape can handle the type - done
        return nodes

    # onnx < opset 8 does not know reshape for other types than float*, wrap the reshape in casts
    input_cast = ctx.insert_new_node_on_input(node, "Cast", node.input[0])
    input_cast.set_attr("to", onnx_pb.TensorProto.FLOAT)
    ctx.copy_shape(name, input_cast.output[0])

    # if the next node is already a cast we don't need to insert another one
    next_nodes = ctx.find_output_consumers(node.output[0])
    if len(next_nodes) != 1 or next_nodes[0].type != "Cast":
        op_name = utils.make_name(node.name)
        output_cast = ctx.insert_new_node_on_output("Cast", node.output[0], name=op_name)
        output_cast.set_attr("to", node.dtype)
        ctx.copy_shape(name, output_cast.output[0])
        nodes.append(output_cast)
    return [input_cast] + nodes


def reshape_op(ctx, node, name, args):
    # T output = Reshape(T tensor, Tshape shape, @type Tshape)
    # T reshaped = Reshape(T data, @INTS shape) - but takes a optional 2nd input for shape
    shape_node = node.inputs[1]
    shape = shape_node.get_tensor_value()
    if shape is None:
        log.error("Reshape on node %s does not have a const shape", node.name)
        return None
    ctx.remove_input(node, node.input[1])
    node.set_attr("shape", shape)
    ctx.set_shape(node.output[0], shape)
    return node


def minmax_op(ctx, node, name, args):
    # tensorflow minimum/maximum support broadcast. Onnx <= opset 7 does not.
    # inject a add(0) as 'broadcast' operator if needed.
    shapeo = ctx.get_shape(node.output[0])
    needs_broadcast_op = []
    for i, name in enumerate(node.input):
        if ctx.get_shape(name) != shapeo:
            needs_broadcast_op.append(i)
    if needs_broadcast_op:
        new_nodes = []
        for i in needs_broadcast_op:
            input_node = node.inputs[i]
            # we don't track dtype for inserted ops but we know the output dtype here is
            # the one we want.
            dtype = ctx.get_dtype(node.output[0])
            zero_name = utils.make_name(input_node.name)
            ctx.make_const(zero_name, "Const", np.zeros(shapeo, dtype=utils.ONNX_TO_NUMPY_DTYPE[dtype]))
            op_name = utils.make_name(input_node.name)
            output_name = op_name + ":0"
            add_node = Node(helper.make_node("Add", [input_node.output[0], zero_name],
                                             [output_name], name=op_name), ctx)
            node.input[i] = output_name
            new_nodes.append(add_node)
        new_nodes.append(node)
        return new_nodes
    return node


def pack_op(ctx, node, name, args):
    # hack to make up for the missing onnx pack op
    axis = node.get_attr("axis").i
    nodes = []
    inputs = []
    # insert Unsqueeze on each input
    for i, n in enumerate(node.inputs):
        op_name = utils.make_name(node.name)
        output_name = op_name + ":0"
        new_node = Node(helper.make_node("Unsqueeze", [node.input[i]], [output_name], name=op_name, axes=[axis]), ctx)
        node.input[i] = output_name
        nodes.append(new_node)
        inputs.append(output_name)
    # concat all unqueezes
    op_name = utils.make_name(node.name)
    output_name = op_name + ":0"
    concat = Node(helper.make_node("Concat", inputs, [output_name], name=op_name, axis=axis), ctx)
    ctx.copy_shape(node.output[0], concat.output[0])
    ctx.replace_all_inputs(ctx.get_nodes(), node.output[0], output_name)
    return [concat] + nodes


def unpack_op(ctx, node, name, args):
    # hack to make up for the missing onnx unpack op
    axis = node.get_attr("axis").i
    # split the tensor into n outputs
    node.type = "Split"
    nodes = [node]
    # for each output we need to squeeze axis
    for i, n in enumerate(node.output):
        op_name = utils.make_name(node.name)
        output_name = op_name + ":" + str(i)
        new_node = Node(helper.make_node("Squeeze", [n], [output_name], name=op_name, axes=[axis]), ctx)
        nodes.append(new_node)
        ctx.copy_shape(n, output_name)
        ctx.replace_all_inputs(ctx.get_nodes(), n, output_name)
    return nodes


def onehot_op(ctx, node, name, args):
    # until there is no onehot op in onnx, a workaround using gather from eye
    indices_name = node.input[0]
    indices_shape = ctx.get_shape(indices_name)
    if len(indices_shape) != 1:
        # TODO: this works for rank=1 but tensorflow supports more than this.
        # Same principle should work but we need to implemtn our own eye.
        raise ValueError("onehot op: only rank1 is supported")
    axis = node.get_attr("axis")
    # axis becomes axis for gather
    node.set_attr("axis", 0)
    depth = node.inputs[1].get_tensor_value()[0]
    on = node.inputs[2].get_tensor_value()[0]
    off = node.inputs[3].get_tensor_value()[0]
    dtype = node.inputs[2].get_tensor_type()
    eye = np.eye(depth, dtype=dtype)
    if on != 0:
        eye[eye == 1] = on
        eye[eye == 0] = off
    else:
        eye[eye == 0] = off
        eye[eye == 1] = on
    const_name = utils.make_name(node.name)
    ctx.make_const(const_name, "Const", eye)
    # setup gather inputs
    del node.input[:]
    node.input.append(const_name)
    node.input.append(indices_name)
    node.type = "Gather"
    if axis.i == 0:
        # TODO: revisit for rank > 1
        name = utils.make_name(node.name)
        transpose_op = ctx.insert_new_node_on_output("Transpose", node.output[0], name)
        ctx.copy_shape(node.output[0], transpose_op.output[0])
        return [node, transpose_op]
    return node


def fused_batchnorm_op7(ctx, node, name, args):
    node.type = "BatchNormalization"
    # tf inputs: x, scale, bias, mean, variance
    # tf outputs: y, batch_mean, batch_var
    # a: data_format, epsilon, is_training
    # onnx inputs: X, scale, B, mean, variance, attributes: epsilon, momentum=0.9, spatial : 1
    # output: mean, var, savedmean, savedvar,
    is_test = node.get_attr("is_training")
    if is_test.i == 0:
        output_len = len(node.output)
        for i in range(output_len - 1):
            del node.output[1]
    return node


def batchnorm_op7(ctx, node, name, args):
    return node


# pylint: enable=W0613,C0111,W0612

# map tensorflow ops to onnx ops. The format below is
# "TFOP": func_to_map, ["OnnxOp", ...]
#
_OPSET_7 = {
    # "Tile": (tile_op7, []),
    # "Sub": (broadcast_op7, []),
    # "RealDiv": (broadcast_op7, ["Div"]),
    # "LogicalAnd": (broadcast_op7, ["And"]),
    # "LogicalOr": (broadcast_op7, ["Or"]),
    # "Greater": (broadcast_op7, []),
    # "Less": (broadcast_op7, []),
    # "Pow": (direct_op, []),
    # "Cast": (direct_op, []),
    # "Acos": (direct_op, []),
    # "Asin": (direct_op, []),
    # "Atan": (direct_op, []),
    # "Cos": (direct_op, []),
    # "Sin": (direct_op, []),
    # "Tan": (direct_op, []),
    # "Multinomial": (multinomial_op, []),
    # "ResizeBilinear": (resizebilinear_op, []),
    "ResizeNearestNeighbor": (upsample_op7, []),
    "Mul": (broadcast_op7, []),
    "BiasAdd": (biasadd_op7, []),
    "BiasAddV1": (biasadd_op7, []),
    "Add": (broadcast_op7, []),
    "FusedBatchNorm": (fused_batchnorm_op7, []),
    "BatchNormalization": (batchnorm_op7, ["BatchNormalization"]),
}
_OPSET_5 = {
    "Reshape": (reshape_op5, []),
    # "ExpandDims": (expanddims_op7, []),
    # "OneHot": (onehot_op, []),
}

_OPSET_6 = {
    "AddN": (direct_op, ["Sum"]),
}

_OPSET_4 = {
    "Conv2D": (conv_op, ["Conv"]),
    "AvgPool": (pool_op, ["AveragePool"]),
    "Sigmoid": (direct_op, []),
    "BiasAdd": (biasadd_op, []),
    "BiasAddV1": (biasadd_op, []),
    "Pad": (pad_op, []),
    "PadV2": (pad_op, ["Pad"]),
    "Placeholder": (placeholder_op, []),
    "PlaceholderV2": (placeholder_op, []),
    "PlaceholderWithDefault": (placeholder_op, []),
    # "ExpandDims": (expanddims_op, []),
    "DepthwiseConv2d": (depthwiseconv_op, []),
    "DepthwiseConv2dNative": (depthwiseconv2dnative_op, []),
    "Concat": (concat_op, ["Concat"]),
    "ConcatV2": (concat_op, ["Concat"]),
    "Const": (const_op, []),
    "ConstV2": (const_op, []),
    "Elu": (direct_op, []),
    "Flatten": (direct_op, []),
    "MaxPool": (pool_op, ["MaxPool"]),
    "MaxPoolV2": (pool_op, ["MaxPool"]),
    "Tanh": (direct_op, []),
    "SpaceToDepth": (spacetodepth_op, []),
    "Softmax": (direct_op, ["Softmax"]),
    "Relu": (direct_op, ["Relu"]),
    "Relu6": (new_relu6_op, []),
    "Reshape": (reshape_op, ["Reshape"]),
    "Add": (broadcast_op, []),
    "Transpose": (transpose_op, []),
    "Squeeze": (squeeze_op, []),
    "Mean": (reduce_op, ["ReduceMean"]),
    "Identity": (identity_op, ["Identity"]),
    "LeakyRelu": (leaky_relu_op, []),
    "Maximum": (minmax_op, ["Max"]),
    "Mul": (broadcast_op, []),
    "Rsqrt": (rsqrt_op, []),
    "Sub": (broadcast_op, []),
    "MatMul": (direct_op, ["MatMul"]),
    "Dense": (dense_op, []),
    "Shape": (direct_op, []),
    "StridedSlice": (stridedslice_op, []),
    "Pack": (pack_op, []),
    # "Max": (reduce_op, ["ReduceMax"]),
    # "Sum": (reduce_op, ["ReduceSum"]),
    "Conv2DBackpropInput": (convtranspose_op, ["ConvTranspose"]),
    "Abs": (direct_op, []),
    "Slice": (slice_op, []),
    # "MirrorPad": (pad_op, ["Pad"]),
    # "Neg": (direct_op, []),
    # "NoOp": (no_op, []),
    # "NotEqual": (direct_op, ["Not"]),
    # "Pow": (pow_op, []),
    # "Prod": (reduce_op, ["ReduceProd"]),
    # "RandomNormal": (direct_op, []),
    # "RandomUniform": (direct_op, []),
    # "RealDiv": (broadcast_op, ["Div"]),
    # "Reciprocal": (direct_op, []),
    # "Size": (direct_op, []),
    # "AvgPool3D": (pool_op, ["AveragePool"]),
    # "ArgMax": (arg_minmax_op, []),
    # "ArgMin": (arg_minmax_op, []),
    # "SplitV": (splitv_op, ["Split"]),
    # "Split": (split_op, ["Split"]),
    # "Sqrt": (direct_op, []),
    # "Square": (square_op, []),
    # "SquaredDifference": (squareddifference_op, []),
    # "StopGradient": (identity_op, ["Identity"]),
    # "Unpack": (unpack_op, []),
    # "Gather": (direct_op, ["Gather"]),
    # "GatherV2": (gatherv2_op, ["Gather"]),
    # "Greater": (broadcast_op, []),
    # "Less": (broadcast_op, []),
    # "Log": (direct_op, []),
    # "LRN": (lrn_op, []),
    # "LogicalAnd": (broadcast_op, ["And"]),
    # "LogicalOr": (broadcast_op, ["Or"]),
    # "Cast": (cast_op, []),
    # "Conv3D": (conv_op, ["Conv"]),
    # "Equal": (broadcast_op, []),
    # "Dropout": (direct_op, []),
    # "Exp": (direct_op, []),
    # "Floor": (direct_op, []),
    # "Min": (reduce_op, ["ReduceMin"]),
    # "Minimum": (minmax_op, ["Min"]),
    # "TopKV2": (topk_op, []),
}

_OPSETS = [
    (4, _OPSET_4),
    (5, _OPSET_5),
    (6, _OPSET_6),
    (7, _OPSET_7),
]


def rewrite_random_uniform(g, ops):
    pattern = \
        OpTypePattern('Add', name='output', inputs=[
            OpTypePattern('Mul', inputs=[
                OpTypePattern('RandomUniform', name='input1'),
                OpTypePattern('Sub', name='input2', inputs=["*", "*"]),
            ]), None
        ])

    matcher = GraphMatcher(pattern)
    match_results = list(matcher.match_ops(ops))
    for match in match_results:
        input2 = match.get_op('input2')
        output = match.get_op('output')
        # max is on input 0
        tmax = input2.inputs[0].get_tensor_value()[0]
        tmin = input2.inputs[1].get_tensor_value()[0]
        shape = g.get_shape(output.output[0])
        dtype = output.dtype
        op_name = utils.make_name("RandomUniform")
        out_name = op_name + ":0"
        new_node = Node(helper.make_node("RandomUniform", [], [out_name],
                                         name=op_name, low=tmin, high=tmax,
                                         dtype=dtype, shape=shape), g)
        ops = g.replace_subgraph(ops, match, [], [output], [], [new_node])

    return ops


def rewrite_transpose(g, ops):
    pattern = \
        OpTypePattern('Transpose', name='output', inputs=[
            OpTypePattern(None),
            OpTypePattern('Sub', inputs=[
                OpTypePattern('Sub', inputs=["*", "*"]),
                OpTypePattern('Range', inputs=["*", "*", "*"]),
            ]),
        ])

    matcher = GraphMatcher(pattern)
    match_results = list(matcher.match_ops(ops))
    for match in match_results:
        output = match.get_op('output')
        shape = g.get_shape(output.input[0])
        dims = [i for i in range(len(shape) - 1, -1, -1)]
        output.set_attr("perm", dims)
        g.remove_input(output, output.input[1])
        ops = g.replace_subgraph(ops, match, [], [], [], [])
        ops.append(output)
    return ops


def rewrite_random_normal(g, ops):
    pattern = \
        OpTypePattern('Add', name='output', inputs=[
            OpTypePattern('Mul', name='input2', inputs=[
                OpTypePattern('RandomStandardNormal', name='input1', inputs=["*"]), "*"
            ]), "*"
        ])

    matcher = GraphMatcher(pattern)
    match_results = list(matcher.match_ops(ops))
    for match in match_results:
        output = match.get_op('output')
        mean = output.inputs[1].get_tensor_value()[0]
        shape = g.get_shape(output.output[0])
        dtype = output.dtype
        op_name = utils.make_name("RandomNormal")
        out_name = op_name + ":0"
        new_node = Node(helper.make_node("RandomNormal", [], [out_name],
                                         name=op_name, shape=shape, mean=mean, scale=1.0,
                                         dtype=dtype), g)
        ops = g.replace_subgraph(ops, match, [], [output], [], [new_node])

    return ops


def rewrite_dropout(g, ops):
    pattern = \
        OpTypePattern('Mul', name='outputs', inputs=[
            OpTypePattern('RealDiv', name="input2"),
            OpTypePattern('Floor', inputs=[
                OpTypePattern('Add', inputs=[
                    OpTypePattern(None, name="input3"),
                    OpTypePattern('RandomUniform'),
                ])
            ]),
        ])
    matcher = GraphMatcher(pattern)
    match_results = list(matcher.match_ops(ops))
    for match in match_results:
        inputs2 = match.get_op('input2')
        outputs = match.get_op('outputs')
        op_name = utils.make_name("Dropout")
        out_name = op_name + ":0"
        new_node = Node(helper.make_node("Dropout", [inputs2.input[0]], [out_name], name=op_name, ratio=1.0), g)
        ops = g.replace_subgraph(ops, match, [inputs2], [outputs], [new_node], [new_node])

    return ops


def rewrite_leaky_relu(g, ops):
    print("---------rewrite_leaky_relu----------")
    pattern = \
        OpTypePattern('Maximum', name='outputs', inputs=[
            OpTypePattern('Mul', name='input2', inputs=[OpTypePattern(None), "*"]), "*"])
    matcher = GraphMatcher(pattern)
    match_results = list(matcher.match_ops(ops))

    for match in match_results:
        inputs2 = match.get_op('input2')
        alpha = inputs2.inputs[0].get_tensor_value()[0]
        outputs = match.get_op('outputs')
        op_name = utils.make_name("LeakyRelu")
        out_name = op_name + ":0"
        new_node = Node(helper.make_node("LeakyRelu", [inputs2.input[1]], [out_name], name=op_name, alpha=alpha), g)
        ops = g.replace_all_inputs(ops, outputs.output[0], out_name)
        to_be_removed = [node for node in match.get_nodes() if node.type != "FusedBatchNorm"]
        for i in range(len(ops) - 1, -1, -1):
            if ops[i] in to_be_removed:
                del ops[i]
        ops.append(new_node)

    return ops


# inception v3 v1 ok
def rewrite_batchnorm(g, ops):
    """
    将 基本运算 合成 BatchNorm 这里面没有 Gamma/scale
    :param g:
    :param ops:
    :return: BatchNormalization
    """
    print("-----------------rewrite_batchnorm----------")
    pattern = \
        OpTypePattern('Add', name='outputs', inputs=[
            OpTypePattern('Mul', name='input2', inputs=[
                "*", OpTypePattern("Rsqrt", name="input4", inputs=[
                    OpTypePattern("Add", name="moving_variance", inputs=["*", "*"])])]),
            OpTypePattern("Sub", name="beta", inputs=["*",
                                                      OpTypePattern("Mul", name="mean", inputs=["*", "*"])])])

    matcher = GraphMatcher(pattern)
    match_results = list(matcher.match_ops(ops))
    for match in match_results:
        inputs2 = match.get_op('input2')
        beta = match.get_op('beta')
        moving_variance = match.get_op('moving_variance')
        mean = match.get_op('mean')
        outputs = match.get_op('outputs')
        op_name = utils.make_name("BatchNormalization")
        out_name = op_name + ":0"
        new_node = Node(
            helper.make_node("BatchNormalization",
                             [inputs2.input[0], beta.input[0], mean.input[0], moving_variance.input[0]], [out_name],
                             name=op_name, epsilon=0.001), g)
        ops = g.replace_all_inputs(ops, outputs.output[0], out_name)
        to_be_removed = [node for node in match.get_nodes() if node.type != "Conv2D" and node.type != "Identity"]
        for i in range(len(ops) - 1, -1, -1):
            if ops[i] in to_be_removed:
                del ops[i]
        ops.append(new_node)

    return ops


# mobilenet
def rewrite_batchnorm1(g, ops):
    """
      将 基本运算 合成 BatchNorm 含有 Gamma/scale
      :param g:
      :param ops:
      :return: BatchNormalization
      """
    print("-----------------rewrite_batchnorm---111--------")

    pattern = \
        OpTypePattern('Add', name='outputs', inputs=[
            OpTypePattern('Mul', name='input2', inputs=[
                "*", OpTypePattern("Mul", name="gamma", inputs=[
                    OpTypePattern("Rsqrt", name="input4", inputs=[
                        OpTypePattern("Add", name="moving_variance", inputs=["*", "*"])]), "*"])]),
            OpTypePattern("Sub", name="beta", inputs=["*",
                                                      OpTypePattern("Mul", name="mean", inputs=["*", "*"])])])

    matcher = GraphMatcher(pattern)
    match_results = list(matcher.match_ops(ops))

    for match in match_results:
        inputs2 = match.get_op('input2')
        gamma = match.get_op('gamma')
        # print("--gamma.input[0]:{}, gamma.input[1]:{}\n".format(gamma.input[0], gamma.input[1]))
        beta = match.get_op('beta')
        moving_variance = match.get_op('moving_variance')
        mean = match.get_op('mean')
        outputs = match.get_op('outputs')
        op_name = utils.make_name("BatchNormalization")
        out_name = op_name + ":0"
        new_node = Node(
            helper.make_node("BatchNormalization",
                             [inputs2.input[0], gamma.input[1], beta.input[0], mean.input[0], moving_variance.input[0]],
                             [out_name], name=op_name, epsilon=0.001), g)
        ops = g.replace_all_inputs(ops, outputs.output[0], out_name)

        to_be_removed = [node for node in match.get_nodes() if
                         node.type != "Conv2D"
                         and node.type != "Identity"
                         and node.type != "DepthwiseConv2dNative"]
        for i in range(len(ops) - 1, -1, -1):
            if ops[i] in to_be_removed:
                del ops[i]
        ops.append(new_node)
    return ops


def rewrite_dense(g, ops):
    print("-------------rewrite_dense-----------------")

    pattern = \
        OpTypePattern('BiasAdd', name='outputs', inputs=[
            OpTypePattern('MatMul', name="input2", inputs=["*", "*"]), "*"])
    matcher = GraphMatcher(pattern)
    match_results = list(matcher.match_ops(ops))
    for match in match_results:
        inputs2 = match.get_op('input2')
        # print("--inputs2.input[0]:{}, inputs2.input[1]:{}\n".format(inputs2.input[0], inputs2.input[1]))
        outputs = match.get_op('outputs')
        op_name = utils.make_name("Dense")
        out_name = op_name + ":0"
        new_node = Node(
            helper.make_node("Dense", [inputs2.input[0], inputs2.input[1], outputs.input[1]], [out_name], name=op_name),
            g)
        ops = g.replace_all_inputs(ops, outputs.output[0], out_name)
        to_be_removed = [node for node in match.get_nodes() if node.type != "Identity"
                         and node.type != "Reshape"
                         and node.type != "Flatten"
                         and node.type != "Const"]
        for i in range(len(ops) - 1, -1, -1):
            if ops[i] in to_be_removed:
                del ops[i]
        ops.append(new_node)
    return ops


def rewrite_flatten(g, ops):
    print("----------------rewrite_flatten---------------")
    pattern = \
        OpTypePattern('Reshape', name='outputs',
                      inputs=["*", OpTypePattern("Pack", name="input2", inputs=[
                          OpTypePattern("StridedSlice", name="slice", inputs=[
                              OpTypePattern("Shape", inputs=["*"]), "*", "*", "*"]), "*"])
                              ])
    matcher = GraphMatcher(pattern)
    match_results = list(matcher.match_ops(ops))
    for match in match_results:
        inputs2 = match.get_op('input2')
        outputs = match.get_op('outputs')
        op_name = utils.make_name("Flatten")
        out_name = op_name + ":0"
        new_node = Node(helper.make_node("Flatten", [outputs.input[0]], [out_name], name=op_name), g)
        ops = g.replace_all_inputs(ops, outputs.output[0], out_name)
        to_be_removed = [node for node in match.get_nodes() if node.type != "Mean"]
        for i in range(len(ops) - 1, -1, -1):
            if ops[i] in to_be_removed:
                del ops[i]
        ops.append(new_node)
    return ops

def delete_ops(g, ops):
    print("----------------rewrite_delete_ops---------------")
    pattern = \
        OpTypePattern('StridedSlice', name='outputs',
                      inputs=[OpTypePattern("Shape", name="input2", inputs=["*"]),
                              "*", "*", "*"])
    matcher = GraphMatcher(pattern)
    match_results = list(matcher.match_ops(ops))
    for match in match_results:
        to_be_removed = [node for node in match.get_nodes() if node.type != "Placeholder" and node.type!="Const"]
        print("to_be_removed:", to_be_removed)
        for i in range(len(ops) - 1, -1, -1):
            if ops[i] in to_be_removed:
                del ops[i]
    return ops




def tensorflow_onnx_mapping(g, continue_on_error, custom_op_handlers):
    print("tensorflow_onnx_mapping......")
    mapped_op = collections.Counter()
    unmapped_op = collections.Counter()

    # create ops mapping for the desired opset
    ops_mapping = {}

    print("_OPSETS......")
    for target_opset, op_map in _OPSETS:
        if target_opset <= g.opset:
            ops_mapping.update(op_map)

    # apply custom ops on top of the assembled opset. We can either completment the opset
    # or override existing ops with a custom op.
    if custom_op_handlers is not None:
        custom_opset = {k: [v, []] for k, v in custom_op_handlers.items()}
        ops_mapping.update(custom_opset)

    ops = g.get_nodes()
    onnx_nodes = []
    for node in ops:
        op = node.type
        map_info = ops_mapping.get(op)

        if map_info is None:
            if continue_on_error:
                unmapped_op[op] += 1
                onnx_nodes.append(node)
                continue
            else:
                raise ValueError("tensorflow op " + op + " is not supported")
        mapped_op[op] += 1
        func, args = map_info
        if args:
            node.type = args[0]
            args = args[1:]
        try:
            onnx_node = func(g, node, node.name, args)

        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            ex_ext = traceback.format_exception(type_, value_, traceback_)
            if continue_on_error:
                print(ex_ext)
                onnx_nodes.append(node)
            else:
                raise ex
        if onnx_node:
            if isinstance(onnx_node, list):
                onnx_nodes.extend(onnx_node)
            else:
                onnx_nodes.append(onnx_node)

    g.set_nodes(onnx_nodes)

    print("tensorflow_onnx_mapping...... done")
    return mapped_op, unmapped_op


def tf_optimize(sess, inputs, outputs, graph_def):
    # print("tf_optimize begin")
    # #"fold_constants(ignore_errors=true)", 注释掉这行可以对 FBN 进行修改
    """Optimize tensorflow graph for inference."""
    transforms = [
        # "fold_constants(ignore_errors=true)",
        "fold_batch_norms",
        "fold_old_batch_norms",
        "sort_by_execution_order",

    ]
    needed_names = [utils.node_name(i) for i in inputs] + [utils.node_name(i) for i in outputs]
    print("---------------needed_names:", needed_names)
    graph_def = graph_util.extract_sub_graph(graph_def, needed_names)
    graph_def = TransformGraph(graph_def, inputs, outputs, transforms)
    return graph_def


def process_tf_graph(tf_graph, continue_on_error=False, verbose=False, target=None,
                     opset=None, custom_op_handlers=None, custom_rewriter=None,
                     extra_opset=None, shape_override=None,inputs=None, middle_inputs=None):
    print("process_tf_graph")
    """Convert tensorflow graph to onnx graph.
        Args:
            tf_graph: tensorflow graph
            continue_on_error: if an op can't be processed (aka there is no mapping), continue
            verbose: print summary stats
            target: list of workarounds applied to help certain platforms
            opset: the opset to be used (int, default is latest)
            custom_op_handlers: dictionary of custom ops handlers
            custom_rewriter: list of custom graph rewriters
        Return:
            onnx graph
    """

    def topological_sort(ops):
        if not continue_on_error:
            g.topological_sort(ops)
        else:
            try:
                g.topological_sort(ops)
            except:
                # if we continue on error, ignore graph cycles so we can report all missing ops
                pass

    if shape_override is None:
        shape_override = {}
    if target is None:
        target = DEFAULT_TARGET

    onnx_nodes, op_cnt, attr_cnt, output_shapes, dtypes = tensorflow_to_onnx(tf_graph, shape_override,inputs=inputs, middle_inputs=middle_inputs)
    print("!!! len(onnx_nodes):", len(onnx_nodes))
    g = Graph(onnx_nodes, output_shapes, dtypes, target, opset, extra_opset)
    ops = g.get_nodes()

    # rewrite graph
    rewriters = [rewrite_transpose, rewrite_flatten, rewrite_random_uniform,
                 rewrite_random_normal, rewrite_dropout, rewrite_leaky_relu,
                 rewrite_batchnorm, rewrite_batchnorm1, rewrite_dense, delete_ops]
    if custom_rewriter is not None:
        rewriters.extend(custom_rewriter)
    for rewrite in rewriters:
        ops = rewrite(g, ops)
        g.set_nodes(ops)

    topological_sort(g.get_nodes())

    if custom_op_handlers is None:
        custom_op_handlers = {}

    mapped_op, unmapped_op = tensorflow_onnx_mapping(g, continue_on_error, custom_op_handlers)

    topological_sort(g.get_nodes())

    g.update_proto()

    if verbose:
        print("tensorflow ops: {}".format(op_cnt))
        print("tensorflow attr: {}".format(attr_cnt))
        print("onnx mapped: {}".format(mapped_op))
        print("onnx unmapped: {}".format(unmapped_op))
    return g
