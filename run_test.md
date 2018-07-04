python -m tf2onnx.convert --input tests/models/fc-layers/frozen.pb --inputs X:0 --outputs output:0 --output tests/models/fc-layers/model.onnx

/home/nishome/jjzhao/github/tensorflow-onnx


ae0_test.yaml   fc_test.yaml              run_pretrained_models.py    test_graph.py
beach.jpg       googlenet_resnet_v2.yaml  run_pretrained_models.yaml  test_internals.py
conv_test.yaml  models                    test_backend.py             unity.yaml


inception_v3_slim :

python tests/run_pretrained_models.py --backend caffe2 --config tests/inception_v3_slim.yaml --onnx-file inception_v3_slim  OK

fc:

python tests/run_pretrained_models.py --backend caffe2 --config tests/fc_test.yaml --onnx-file fc    OK 

conv:

python tests/run_pretrained_models.py --backend caffe2 --config tests/conv_test.yaml --onnx-file conv   OK


googlenet_resnet_v2:

python tests/run_pretrained_models.py --backend caffe2 --config tests/googlenet_resnet_v2.yaml --onnx-file googlenet_resnet_v2 OK


googlenet_v1_slim：


python tests/run_pretrained_models.py --backend caffe2 --config tests/googlenet_v1_slim.yaml --onnx-file googlenet_v1_slim  OK



googlenet_v4_slim：


python tests/run_pretrained_models.py --backend caffe2 --config tests/googlenet_v4_slim.yaml --onnx-file googlenet_v4_slim   OK

mobilenet_v1_75_192：

python tests/run_pretrained_models.py --backend caffe2 --config tests/mobilenet_v1_75_192.yaml  --onnx-file mobilenet_v1_75_192 OK


mobilenet_v1_100_224：

python tests/run_pretrained_models.py --backend caffe2 --config tests/mobilenet_v1_100_224.yaml  --onnx-file mobilenet_v1_100_224  OK
---

vgg16:


python tests/run_pretrained_models.py --backend caffe2 --config tests/vgg16.yaml  --onnx-file vgg16   OK

is_training = False       479  1000   481 car wheel

不用做图片处理
---
resnet_v2_50:

python tests/run_pretrained_models.py --backend caffe2 --config tests/resnet_v2_50.yaml  --onnx-file resnet_v2_50

/ 127.5 -1  需要做图片处理  这里的输出 是 5 维的 

caffe2 error:

warnings.warn("This version of onnx-caffe2 targets ONNX operator set version {}, but the model we are trying to import uses version {}.  We will try to import it anyway, but if the model uses operators which had BC-breaking changes in the intervening versions, import will fail.".format(cls._known_opset_version, imp.version))
ONNX FATAL: [enforce fail at backend.cc:653] . Caffe2 only supports padding 2D Tensor, whereas padding is [0, 3, 3, 0, 0, 3, 3, 0, ] 
ONNX FATAL: [enforce fail at backend.cc:653] . Caffe2 only supports padding 2D Tensor, whereas padding is [0, 1, 1, 0, 0, 1, 1, 0, ] 
ONNX FATAL: [enforce fail at backend.cc:653] . Caffe2 only supports padding 2D Tensor, whereas padding is [0, 1, 1, 0, 0, 1, 1, 0, ] 
ONNX FATAL: [enforce fail at backend.cc:653] . Caffe2 only supports padding 2D Tensor, whereas padding is [0, 1, 1, 0, 0, 1, 1, 0, ] 
	run_onnx FAIL ONNX conversion failed
:x
---
resnet_v1_50:

python tests/run_pretrained_models.py --backend caffe2 --config tests/resnet_v1_50.yaml  --onnx-file resnet_v1_50


tf2onnx.convert:

python -m tf2onnx.convert --input frozen_resnet_v1_50.pb --inputs input:0 --outputs resnet_v1_50/logits/BiasAdd:0 --output resnet_v1_50_model.onnx
python -m tf2onnx.convert --input frozen_resnet_v1_50.pb --inputs input:0 --outputs resnet_v1_50/logits/BiasAdd:0 --output resnet_v1_no_fu.onnx


   







不做图像归一化处理，5 维的 [0][0][0] 才能输出 index   1000 类  479 +2 481 carwheel

换一种 freeze graph 的方式，
frzee_graph:


resnet_v2_50:
  model: tests/models/resnet_v2_50/frozen_resnet_v2_50.pb
  input_get: get_beach
  inputs:
    "input:0": [1, 224, 224, 3]
  outputs:
    - resnet_v2_50/logits/BiasAdd:0


sudo python -m tensorflow.python.tools.freeze_graph \
    --input_graph=pb_model/model.pb \
    --input_binary=false \
    --input_names=input:0 \
    --output_node_names=resnet_v2_50/logits/BiasAdd \
    --input_checkpoint=resnet_v2_50.ckpt \
    --output_graph=frozen_graph.pb



--------------------------------------------
mobilenet_v1_75_192：


run_test begin
run_test_name: mobilenet_v1_75_192
	downloaded tests/models/mobilenetV1_192/mobilenet_v1_0.75_192/frozen_graph.pb
name: input:0 shape: [1, 192, 192, 3]
resize_to: [192, 192]
run_tensorflow(): so we have a reference output
type(result.shape: (1, 1001)
	 -----np.argmax(result): [480]
!!! tf_results:
 [array([[2.7309464e-11, 9.2483560e-10, 4.6631465e-10, ..., 3.9673542e-11,
        9.6685188e-11, 3.4415204e-09]], dtype=float32)]
	 !!! wow tensorflow OK
	 wow to_onnx OK
	created mobilenet_v1_75_192/mobilenet_v1_75_192.onnx
backend==caffe2
run_caffe2()
/data1/train_disk1/Train_tools/pytorch/build/caffe2/python/onnx/backend.py:728: UserWarning: This version of onnx-caffe2 targets ONNX operator set version 6, but the model we are trying to import uses version 8.  We will try to import it anyway, but if the model uses operators which had BC-breaking changes in the intervening versions, import will fail.
  warnings.warn("This version of onnx-caffe2 targets ONNX operator set version {}, but the model we are trying to import uses version {}.  We will try to import it anyway, but if the model uses operators which had BC-breaking changes in the intervening versions, import will fail.".format(cls._known_opset_version, imp.version))
prepared_backend: <caffe2.python.onnx.backend_rep.Caffe2Rep object at 0x7ff398148f60>
WARNING: Logging before InitGoogleLogging() is written to STDERR
W0628 19:58:09.210032 23843 init.h:99] Caffe2 GlobalInit should be run before any other API calls.
W0628 19:58:09.210464 23843 init.h:99] Caffe2 GlobalInit should be run before any other API calls.
	 onnx_results(run_caffe2):
 Outputs(_0=array([[2.7309266e-11, 9.2482894e-10, 4.6630771e-10, ..., 3.9673709e-11,
        9.6684855e-11, 3.4414822e-09]], dtype=float32))
	 wow run_onnx OK
	 np.testing.assert_allclose:  两个对象的近似程度 指定的容差限,
onnx_results.argmax: 480
tf_result[i]: [[2.7309464e-11 9.2483560e-10 4.6631465e-10 ... 3.9673542e-11
  9.6685188e-11 3.4415204e-09]] 
 onnx_results[i]: [[2.7309266e-11 9.2482894e-10 4.6630771e-10 ... 3.9673709e-11
  9.6684855e-11 3.4414822e-09]]
	 wow  Results: OK
=== RESULT: 0 failed of 1, backend=caffe2




inceptionV1:

run_test begin
run_test_name: googlenet_v1_slim
	downloaded tests/models/inceptionV1/inception_v1_2016_08_28_frozen.pb
name: input:0 shape: [1, 224, 224, 3]
resize_to: [224, 224]
run_tensorflow(): so we have a reference output
type(result.shape: (1, 1001)
	 -----np.argmax(result): [480]
!!! tf_results:
 [array([[1.05066807e-04, 3.70625858e-05, 8.58852000e-05, ...,
        1.94694821e-05, 1.18404983e-04, 1.05927415e-04]], dtype=float32)]
	 !!! wow tensorflow OK
	 wow to_onnx OK
	created googlenet_v1_slim/googlenet_v1_slim.onnx
backend==caffe2
run_caffe2()
/data1/train_disk1/Train_tools/pytorch/build/caffe2/python/onnx/backend.py:728: UserWarning: This version of onnx-caffe2 targets ONNX operator set version 6, but the model we are trying to import uses version 8.  We will try to import it anyway, but if the model uses operators which had BC-breaking changes in the intervening versions, import will fail.
  warnings.warn("This version of onnx-caffe2 targets ONNX operator set version {}, but the model we are trying to import uses version {}.  We will try to import it anyway, but if the model uses operators which had BC-breaking changes in the intervening versions, import will fail.".format(cls._known_opset_version, imp.version))
prepared_backend: <caffe2.python.onnx.backend_rep.Caffe2Rep object at 0x7f3a8755da58>
WARNING: Logging before InitGoogleLogging() is written to STDERR
W0628 19:34:27.433046 23260 init.h:99] Caffe2 GlobalInit should be run before any other API calls.
W0628 19:34:27.433540 23260 init.h:99] Caffe2 GlobalInit should be run before any other API calls.
	 onnx_results(run_caffe2):
 Outputs(_0=array([[1.05066989e-04, 3.70626512e-05, 8.58853527e-05, ...,
        1.94695349e-05, 1.18405078e-04, 1.05927706e-04]], dtype=float32))
	 wow run_onnx OK
	 np.testing.assert_allclose:  两个对象的近似程度 指定的容差限,
onnx_results.argmax: 480
tf_result[i]: [[1.05066807e-04 3.70625858e-05 8.58852000e-05 ... 1.94694821e-05
  1.18404983e-04 1.05927415e-04]] 
 onnx_results[i]: [[1.05066989e-04 3.70626512e-05 8.58853527e-05 ... 1.94695349e-05
  1.18405078e-04 1.05927706e-04]]
	 wow  Results: OK
=== RESULT: 0 failed of 1, backend=caffe2






inceptionV4:

run_test begin
run_test_name: googlenet_v4_slim
	downloaded tests/models/inceptionV4/inception_v4_2016_09_09_frozen.pb
name: input:0 shape: [1, 299, 299, 3]
resize_to: [299, 299]
run_tensorflow(): so we have a reference output
type(result.shape: (1, 1001)
	 -----np.argmax(result): [480]
!!! tf_results:
 [array([[1.1814975e-04, 5.9863960e-05, 8.9771624e-05, ..., 9.9724122e-05,
        9.6938231e-05, 7.5561053e-05]], dtype=float32)]
	 !!! wow tensorflow OK
	 wow to_onnx OK
	created googlenet_v4_slim/googlenet_v4_slim.onnx
backend==caffe2
run_caffe2()
/data1/train_disk1/Train_tools/pytorch/build/caffe2/python/onnx/backend.py:728: UserWarning: This version of onnx-caffe2 targets ONNX operator set version 6, but the model we are trying to import uses version 8.  We will try to import it anyway, but if the model uses operators which had BC-breaking changes in the intervening versions, import will fail.
  warnings.warn("This version of onnx-caffe2 targets ONNX operator set version {}, but the model we are trying to import uses version {}.  We will try to import it anyway, but if the model uses operators which had BC-breaking changes in the intervening versions, import will fail.".format(cls._known_opset_version, imp.version))
prepared_backend: <caffe2.python.onnx.backend_rep.Caffe2Rep object at 0x7f534065b080>
WARNING: Logging before InitGoogleLogging() is written to STDERR
W0628 19:46:37.191563 23503 init.h:99] Caffe2 GlobalInit should be run before any other API calls.
W0628 19:46:37.192576 23503 init.h:99] Caffe2 GlobalInit should be run before any other API calls.
	 onnx_results(run_caffe2):
 Outputs(_0=array([[1.1814983e-04, 5.9864004e-05, 8.9771776e-05, ..., 9.9724210e-05,
        9.6938398e-05, 7.5561031e-05]], dtype=float32))
	 wow run_onnx OK
	 np.testing.assert_allclose:  两个对象的近似程度 指定的容差限,
onnx_results.argmax: 480
tf_result[i]: [[1.1814975e-04 5.9863960e-05 8.9771624e-05 ... 9.9724122e-05
  9.6938231e-05 7.5561053e-05]] 
 onnx_results[i]: [[1.1814983e-04 5.9864004e-05 8.9771776e-05 ... 9.9724210e-05
  9.6938398e-05 7.5561031e-05]]
	 wow  Results: OK
=== RESULT: 0 failed of 1, backend=caffe2



mobilenet_v1_100_224：

run_test begin
run_test_name: mobilenet_v1_100_224
	downloaded tests/models/mobilenetV1_224/mobilenet_v1_1.0_224/frozen_graph.pb
name: input:0 shape: [1, 224, 224, 3]
resize_to: [224, 224]
run_tensorflow(): so we have a reference output
type(result.shape: (1, 1001)
	 -----np.argmax(result): [480]
!!! tf_results:
 [array([[3.0019881e-10, 3.0031309e-08, 9.9150771e-11, ..., 9.3352333e-11,
        4.5039053e-10, 5.3988374e-09]], dtype=float32)]
	 !!! wow tensorflow OK
	 wow to_onnx OK
	created mobilenet_v1_100_224/mobilenet_v1_100_224.onnx
backend==caffe2
run_caffe2()
/data1/train_disk1/Train_tools/pytorch/build/caffe2/python/onnx/backend.py:728: UserWarning: This version of onnx-caffe2 targets ONNX operator set version 6, but the model we are trying to import uses version 8.  We will try to import it anyway, but if the model uses operators which had BC-breaking changes in the intervening versions, import will fail.
  warnings.warn("This version of onnx-caffe2 targets ONNX operator set version {}, but the model we are trying to import uses version {}.  We will try to import it anyway, but if the model uses operators which had BC-breaking changes in the intervening versions, import will fail.".format(cls._known_opset_version, imp.version))
prepared_backend: <caffe2.python.onnx.backend_rep.Caffe2Rep object at 0x7fc898b1ff28>
WARNING: Logging before InitGoogleLogging() is written to STDERR
W0628 19:55:26.691885 23684 init.h:99] Caffe2 GlobalInit should be run before any other API calls.
W0628 19:55:26.692329 23684 init.h:99] Caffe2 GlobalInit should be run before any other API calls.
	 onnx_results(run_caffe2):
 Outputs(_0=array([[3.0019906e-10, 3.0031565e-08, 9.9150861e-11, ..., 9.3352062e-11,
        4.5039011e-10, 5.3988938e-09]], dtype=float32))
	 wow run_onnx OK
	 np.testing.assert_allclose:  两个对象的近似程度 指定的容差限,
onnx_results.argmax: 480
tf_result[i]: [[3.0019881e-10 3.0031309e-08 9.9150771e-11 ... 9.3352333e-11
  4.5039053e-10 5.3988374e-09]] 
 onnx_results[i]: [[3.0019906e-10 3.0031565e-08 9.9150861e-11 ... 9.3352062e-11
  4.5039011e-10 5.3988938e-09]]
	 wow  Results: OK



bazel run tensorflow/tools/benchmark:benchmark_model -- --graph=frozen_model_vgg_16.pb --show_flops --input_layer=Placeholder --input_layer_type=float --input_layer_shape=240,240,3 --output_layer=out


yolov2:

python tests/run_pretrained_models.py --backend caffe2 --config tests/yolov2_test.yaml --onnx-file yolov2_test


upyter notebook --no-browser --ip=192.168.200.76 --port=8889


/data1/train_disk1/Train_tools/pytorch/build/caffe2


# Clone Caffe2's source code from our Github repository
git clone --recursive https://github.com/pytorch/pytorch.git && cd pytorch
git submodule update --init

# Create a directory to put Caffe2's build files in
mkdir build && cd build

# Configure Caffe2's build
# This looks for packages on your machine and figures out which functionality
# to include in the Caffe2 installation. The output of this command is very
# useful in debugging.
cmake ..

# Compile, link, and install Caffe2
sudo make install

summarize_graph --in_graph=tests/models/fc-layers/frozen.pb


bazel build tensorflow/tools/graph_transforms:summarize_graph
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=frozen_model_vgg_16.pb



mobilenet:

bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=frozen_graph.pb
Found 1 possible inputs: (name=input, type=float(1), shape=[1,224,224,3]) 
No variables spotted.
Found 1 possible outputs: (name=MobilenetV1/Predictions/Reshape_1, op=Reshape) 
Found 4254920 (4.25M) const parameters, 0 (0) variable parameters, and 0 control_edges
Op types used: 166 Const, 138 Identity, 81 Mul, 54 Add, 27 Relu6, 27 Rsqrt, 27 Sub, 15 Conv2D, 13 DepthwiseConv2dNative, 2 Reshape, 1 AvgPool, 1 BiasAdd, 1 Placeholder, 1 Softmax, 1 Squeeze
To use with tensorflow/tools/benchmark:benchmark_model try these arguments:
bazel run tensorflow/tools/benchmark:benchmark_model -- --graph=frozen_graph.pb --show_flops 
--input_layer=input --input_layer_type=float --input_layer_shape=1,224,224,3 
--output_layer=MobilenetV1/Predictions/Reshape_1


vgg16:

bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=frozen_model_vgg_16.pb
Found 1 possible inputs: (name=Placeholder, type=float(1), shape=[240,240,3]) 
No variables spotted.
Found 1 possible outputs: (name=out, op=Cast) 
2018-06-29 13:06:41.059938: W tensorflow/core/framework/allocator.cc:108] Allocation of 411041792 exceeds 10% of system memory.
2018-06-29 13:06:41.317519: W tensorflow/core/framework/allocator.cc:108] Allocation of 67108864 exceeds 10% of system memory.
Found 138357559 (138.36M) const parameters, 0 (0) variable parameters, and 0 control_edges
Op types used: 41 Const, 32 Identity, 16 BiasAdd, 16 Conv2D, 15 Relu, 5 MaxPool, 4 Add, 4 Mul, 2 Floor, 2 RandomUniform, 2 RealDiv, 2 Sub, 1 Cast, 1 ExpandDims, 1 Placeholder, 1 Squeeze
To use with tensorflow/tools/benchmark:benchmark_model try these arguments:
bazel run tensorflow/tools/benchmark:benchmark_model -- --graph=frozen_model_vgg_16.pb 
--show_flops --input_layer=Placeholder --input_layer_type=float --input_layer_shape=240,240,3 --output_layer=out



resnet_v2_50:


bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=frozen_resnet_v2_50.pb
Found 1 possible inputs: (name=Placeholder, type=float(1), shape=[224,224,3]) 
No variables spotted.
Found 1 possible outputs: (name=output, op=Cast) 
Found 23519395 (23.52M) const parameters, 0 (0) variable parameters, and 0 control_edges
Op types used: 276 Const, 172 Identity, 53 Conv2D, 49 FusedBatchNorm, 49 Relu, 21 BiasAdd, 16 Add, 4 MaxPool, 4 Pad, 1 Cast, 1 ExpandDims, 1 Mean, 1 Placeholder
To use with tensorflow/tools/benchmark:benchmark_model try these arguments:
bazel run tensorflow/tools/benchmark:benchmark_model -- --graph=frozen_resnet_v2_50.pb --show_flops --input_layer=Placeholder --input_layer_type=float --input_layer_shape=224,224,3 --output_layer=output