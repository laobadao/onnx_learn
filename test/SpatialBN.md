github ¹Ù·½µÄ BN node


node {
  input: "gpu_0/res2_0_branch2a_1"
  input: "gpu_0/res2_0_branch2a_bn_s_0"
  input: "gpu_0/res2_0_branch2a_bn_b_0"
  input: "gpu_0/res2_0_branch2a_bn_rm_0"
  input: "gpu_0/res2_0_branch2a_bn_riv_0"
  output: "gpu_0/res2_0_branch2a_bn_1"
  name: ""
  op_type: "BatchNormalization"
  attribute {
    name: "epsilon"
    f: 1.0000000656873453e-05
    type: FLOAT
  }
}



node {
  input: "resnet_v1_50/conv1/Conv2D__3:0"
  input: "resnet_v1_50/conv1/BatchNorm/gamma:0"
  input: "resnet_v1_50/conv1/BatchNorm/beta:0"
  input: "resnet_v1_50/conv1/BatchNorm/moving_mean:0"
  input: "resnet_v1_50/conv1/BatchNorm/moving_variance:0"
  output: "resnet_v1_50/conv1/BatchNorm/FusedBatchNorm:0"
  output: "resnet_v1_50/conv1/BatchNorm/FusedBatchNorm:1"
  output: "resnet_v1_50/conv1/BatchNorm/FusedBatchNorm:2"
  output: "resnet_v1_50/conv1/BatchNorm/FusedBatchNorm:3"
  output: "resnet_v1_50/conv1/BatchNorm/FusedBatchNorm:4"
  name: "resnet_v1_50/conv1/BatchNorm/FusedBatchNorm"
  op_type: "BatchNormalization"
  attribute {
    name: "epsilon"
    f: 1.0009999641624745e-05
    type: FLOAT
  }
}


nique_ptr<OperatorBase> _CreateOperator(
    const OperatorDef& operator_def,
    Workspace* ws) {
  static StaticLinkingProtector g_protector;
  const auto& op_type = operator_def.type();
  const auto& device_type = operator_def.device_option().device_type();

#ifndef CAFFE2_NO_OPERATOR_SCHEMA
  // first, check with OpSchema if the operator is legal.
  auto* schema = OpSchemaRegistry::Schema(op_type);
  if (schema) {
    CAFFE_ENFORCE(
        schema->Verify(operator_def),
        "Operator def did not pass schema checking: ",
        ProtoDebugString(operator_def));
  } else {
    // We would like to recommend every op to register its schema, so if there
    // is not one, we print a LOG_ERROR. But we will still allow the operator
    // to be constructed.
    LOG(ERROR) << "Cannot find operator schema for " << op_type
               << ". Will skip schema checking.";
  }
#endif


python look.py |tee result.txt

Input index 3 (resnet_v1_50/conv1/BatchNorm/moving_mean:0) and output idx 1 (resnet_v1_50/conv1/BatchNorm/FusedBatchNorm:3) 
are not in-place but should be as required by op SpatialBN
WARNING:caffe2.python.workspace:Original python traceback for operator `4` in network `tf2onnx_predict` in exception above (most recent call last):
	run_onnx FAIL [enforce fail at operator.cc:113] schema->Verify(operator_def). 
	Operator def did not pass schema checking: input: "resnet_v1_50/conv1/Conv2D__3:0" 
	input: "resnet_v1_50/conv1/BatchNorm/gamma:0" input: "resnet_v1_50/conv1/BatchNorm/beta:0" 
	input: "resnet_v1_50/conv1/BatchNorm/moving_mean:0" input: "resnet_v1_50/conv1/BatchNorm/moving_variance:0" 
	output: "resnet_v1_50/conv1/BatchNorm/FusedBatchNorm:0" output: "resnet_v1_50/conv1/BatchNorm/FusedBatchNorm:1" 
	output: "resnet_v1_50/conv1/BatchNorm/FusedBatchNorm:2" output: "resnet_v1_50/conv1/BatchNorm/FusedBatchNorm:3" 
	output: "resnet_v1_50/conv1/BatchNorm/FusedBatchNorm:4" name: "resnet_v1_50/conv1/BatchNorm/FusedBatchNorm" 
	type: "SpatialBN" arg { name: "epsilon" f: 1.001e-05 } arg { name: "is_test" i: 1 } device_option { device_type: 0 cuda_gpu_id: 0 } 
=== RESULT: 1 failed of 1, backend=caffe2


E0703 16:02:11.373966 58901 operator_schema.cc:64] Input index 3 (resnet_v1_50/conv1/BatchNorm/moving_mean:0) and output idx 1
(resnet_v1_50/conv1/BatchNorm/FusedBatchNorm:3) are not in-place but should be as required by op SpatialBN
WARNING:caffe2.python.workspace:Original python traceback for operator `4` in network `tf2onnx_predict` in exception above (most recent call last):
	run_onnx FAIL [enforce fail at operator.cc:113] schema->Verify(operator_def). Operator def did not pass schema checking: 
	input: "resnet_v1_50/conv1/Conv2D__3:0" input: "resnet_v1_50/conv1/BatchNorm/gamma:0" input: "resnet_v1_50/conv1/BatchNorm/beta:0" 
	input: "resnet_v1_50/conv1/BatchNorm/moving_mean:0" input: "resnet_v1_50/conv1/BatchNorm/moving_variance:0" 
	output: "resnet_v1_50/conv1/BatchNorm/FusedBatchNorm:0" output: "resnet_v1_50/conv1/BatchNorm/FusedBatchNorm:1"
	output: "resnet_v1_50/conv1/BatchNorm/FusedBatchNorm:2" output: "resnet_v1_50/conv1/BatchNorm/FusedBatchNorm:3" 
	output: "resnet_v1_50/conv1/BatchNorm/FusedBatchNorm:4" name: "resnet_v1_50/conv1/BatchNorm/FusedBatchNorm" type: "SpatialBN" 
	arg { name: "epsilon" f: 1.001e-05 } arg { name: "is_test" i: 1 } device_option { device_type: 0 cuda_gpu_id: 0 } 
=== RESULT: 1 failed of 1, backend=caffe2




/data1/train_disk1/Train_tools/pytorch/build/caffe2/python/onnx/backend.py:728: UserWarning: This version of onnx-caffe2 targets ONNX operator set version 6, but the model we are trying to import uses version 8.  We will try to import it anyway, but if the model uses operators which had BC-breaking changes in the intervening versions, import will fail.
  warnings.warn("This version of onnx-caffe2 targets ONNX operator set version {}, but the model we are trying to import uses version {}.  We will try to import it anyway, but if the model uses operators which had BC-breaking changes in the intervening versions, import will fail.".format(cls._known_opset_version, imp.version))
ONNX FATAL: [enforce fail at backend.cc:653] . 


Caffe2 only supports padding 2D Tensor, whereas padding is [0, 3, 3, 0, 0, 3, 3, 0, ] 
ONNX FATAL: [enforce fail at backend.cc:653] . Caffe2 only supports padding 2D Tensor, whereas padding is [0, 1, 1, 0, 0, 1, 1, 0, ] 
ONNX FATAL: [enforce fail at backend.cc:653] . Caffe2 only supports padding 2D Tensor, whereas padding is [0, 1, 1, 0, 0, 1, 1, 0, ] 
ONNX FATAL: [enforce fail at backend.cc:653] . Caffe2 only supports padding 2D Tensor, whereas padding is [0, 1, 1, 0, 0, 1, 1, 0, ] 
	run_onnx FAIL ONNX conversion failed


def pad_op(ctx, node, name, args):
    # T output = Pad(T input, Tpaddings paddings, @type Tpaddings)
    # T output = Pad(T data, @STRING mode, @INTS pads, @FLOAT value)
    paddings = np.array(node.inputs[1].get_tensor_value()).transpose().flatten()
    print("paddings:", paddings)
    print("node before:", node)
    ctx.remove_input(node, node.input[1])
    node.set_attr("pads", paddings)
    print("node after:", node)
    return node


https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py


@classmethod
   def _create_pad(cls, init_model, pred_model, n, opset_version):
        if opset_version < 2:
            pads = n.attrs['paddings']
        else:
            pads = n.attrs['pads']
        if not (len(pads) == 8 and
                # first two dim is for batch and channel
                set(pads[:2] + pads[4:6]) == {0}):
            raise ValueError('Caffe2 only supports padding 2D Tensor, whereas padding is ' + str(pads))
        # Guard the invalid (negative) pads attribute.
        if min(pads) < 0:
            raise ValueError('ONNX does not support negative pads in Pad, but get {}.'.format(pads))
        pads[:] = pads[2:4] + pads[6:8]
        return cls._common_onnx_node_to_caffe2_op(init_model, pred_model, n, opset_version)


Caffe2Ops Caffe2Backend::CreatePad(OnnxNode* onnx_node, int opset_version) {
  const auto& node = onnx_node->node;
  auto& attributes = onnx_node->attributes;
  ::google::protobuf::RepeatedField<::google::protobuf::int64> pads;
  std::string pad_name = opset_version < 2 ? "paddings" : "pads";
  pads = attributes
             .get<::google::protobuf::RepeatedField<::google::protobuf::int64>>(
                 pad_name);
  std::string str;
  std::stringstream ss;
  ss << "[";
  for (const auto& i : pads) {
    ss << i << ", ";
  }
  ss << "]";
  str = ss.str();

  // Guard the invalid (negative) pads attribute.
  for (const auto i : pads) {
    if (i < 0) {
      CAFFE_THROW("ONNX does not support negative pads in Pad, but get ", str);
    }
  }

  // first two dim is for batch and channel. Note that now all the values are
  // non-negative
  if (!(pads.size() == 8 &&
        (pads.Get(0) + pads.Get(1) + pads.Get(4) + pads.Get(5) == 0))) {
    CAFFE_THROW(
        "Caffe2 only supports padding 2D Tensor, whereas padding is ", str);
  }

  // rewrite the padding info
  auto* attr = attributes.AddRewrittenAttibute(pad_name);
  attr->add_ints(pads.Get(2));
  attr->add_ints(pads.Get(3));
  attr->add_ints(pads.Get(6));
  attr->add_ints(pads.Get(7));

  return CommonOnnxNodeToCaffe2Ops(onnx_node, opset_version);
}


