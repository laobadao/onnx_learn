python -m tf2onnx.convert --input tests/models/fc-layers/frozen.pb --inputs X:0 --outputs output:0 --output tests/models/fc-layers/model.onnx

/home/nishome/jjzhao/github/tensorflow-onnx


ae0_test.yaml   fc_test.yaml              run_pretrained_models.py    test_graph.py
beach.jpg       googlenet_resnet_v2.yaml  run_pretrained_models.yaml  test_internals.py
conv_test.yaml  models                    test_backend.py             unity.yaml


inception_v3_slim :

python tests/run_pretrained_models.py --backend caffe2 --config tests/inception_v3_slim.yaml --onnx-file inception_v3_slim  OK


tf2onnx:


python -m tf2onnx.convert --input inception_v3_2016_08_28_frozen.pb --inputs input:0 --outputs InceptionV3/Predictions/Softmax:0 --output inception_v3.onnx


inceptionv3 tensorflow 1.9 版本：

python -m tf2onnx.convert --input frozen.pb --inputs input:0 --outputs InceptionV3/Logits/Predictions/Softmax:0 --output inception_v3_new.onnx


python tests/run_pretrained_models.py --backend caffe2 --config tests/inceptionv3.yaml --onnx-file inception_v3_new

---------------------------
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


tf2onnx convert:

python -m tf2onnx.convert --input frozen_model_vgg_16.pb --inputs input:0 --outputs vgg_16/fc8/squeezed:0 --output vgg16.onnx

is_training = False       479  1000   481 car wheel

不用做图片处理
---
resnet_v2_50:

python tests/run_pretrained_models.py --backend caffe2 --config tests/resnet_v2_50.yaml  --onnx-file resnet_v2_50  OK


/ 127.5 -1  需要做图片处理  这里的输出 是 5 维的 

tf2onnx.convert:

python -m tf2onnx.convert --input frozen_resnet_v2_50.pb --inputs input:0 --outputs output:0 --output resnet_v2_50_model.onnx


---------------------

yolov2:

python tests/run_pretrained_models.py --backend caffe2 --config tests/yolov2.yaml --onnx-file yolov2


tf2onnx.convert: 


python -m tf2onnx.convert --input frozen_yolov2.pb --inputs input:0 --outputs output_bboxes:0,output_obj:0,output_class:0 --output yolov2_1.onnx




----------------------------------------------------------

yolov3:


tf2onnx.convert: 


python -m tf2onnx.convert --input frozen_yolov3.pb --inputs input:0 --outputs output:0 --output yolov3.onnx



python tests/run_pretrained_models.py --backend caffe2 --config tests/yolov3.yaml --onnx-file yolov3


python -m tensorflow.python.tools.freeze_graph \
    --input_graph=output/export.pb \
    --input_binary=true \
    --input_names=input,istraining,img_hw \
    --output_node_names=output_boxes,output_scores,output_classes \
    --input_checkpoint=ckpt/yolov3.ckpt-99001 \
    --output_graph=yolov3_pb/frozen.pb

----------------------------------------

resnet_v1_50:

python tests/run_pretrained_models.py --backend caffe2 --config tests/resnet_v1_50.yaml  --onnx-file resnet_v1_50  OK


tf2onnx.convert:

python -m tf2onnx.convert --input frozen_resnet_v1_50.pb --inputs input:0 --outputs resnet_v1_50/logits/BiasAdd:0 --output resnet_v1_50_model.onnx
python -m tf2onnx.convert --input frozen_resnet_v1_50.pb --inputs input:0 --outputs resnet_v1_50/logits/BiasAdd:0 --output resnet_v1_no_fu.onnx




不做图像归一化处理，5 维的 [0][0][0] 才能输出 index   1000 类  479 +2 481 carwheel

换一种 freeze graph 的方式，
frzee_graph:

---------------------
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



--------------------------------------------------

ssd_mobile_v1_qu:


run_test:


python tests/run_pretrained_models.py --backend caffe2 --config tests/ssd_mobile.yaml --onnx-file ssd_mobile

python tests/run_pretrained_models.py --backend caffe2 --config tests/ssd_mobile_1.yaml --onnx-file ssd_mobile


python onnx2keras.py ssd_mobile.onnx ssd_mobile.xlsx ssd_mobile.json ssd_mobile.h5


tf2onnx.convert: 


python -m tf2onnx.convert --input frozen_inference_graph.pb --inputs image_tensor:0 --outputs Squeeze:0,concat_1:0 --output ssd_mobile_v1_q.onnx

python -m tf2onnx.convert --input frozen_model.pb --inputs image_tensor:0 --outputs Squeeze:0,concat_1:0 --output ssd_mobile_v1_q.onnx


--------------------------------------------
faster_rcnn_resnet50:


run_test:


python tests/run_pretrained_models.py --backend caffe2 --config tests/faster_rcnn_resnet50.yaml --onnx-file faster_rcnn_resnet50

python tests/run_pretrained_models.py --backend caffe2 --config tests/faster_rcnn_resnet50_2.yaml --onnx-file faster_rcnn_resnet50

---------------------------------


faster-rcnn-101:

python inference.py --data_dir='demos' \
                    --save_dir='inference_results' \
                    --GPU='0'

python inference.py --data_dir='demos'  --save_dir='inference_results'  --GPU='0'

run_test:

python tests/run_pretrained_models.py --backend caffe2 --config tests/faster_rcnn_resnet101.yaml --onnx-file faster_rcnn_resnet101

python -m tf2onnx.convert --input faster_rcnn_resnet101.pb --inputs image_tensor:0 --outputs postprocess_fastrcnn/concat:0,postprocess_fastrcnn/concat_1:0,postprocess_fastrcnn/concat_2:0 --middle_inputs Preprocessor/sub:0 --output faster_rcnn_resnet101.onnx



---------------------------------


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




rscore: 0.9799035
box: [446.71952467698316, 310.1409673690796, 568.1687340369591, 617.913818359375]
cls_names[cls]: person

score: 0.9995659
box: [688.3275146484375, 367.7403926849365, 1152.9457538311299, 646.6171741485596]
cls_names[cls]: bicycle

score: 0.99960583
box: [186.98643845778244, 39.90745544433594, 664.1616492638221, 300.20434856414795]
cls_names[cls]: car

score: 0.99831796
box: [14.052596459021935, 73.91496896743774, 172.0737022986779, 185.18189191818237]
cls_names[cls]: car

score: 0.99984586
box: [84.39684589092548, 321.8587398529053, 329.7792733999399, 499.8103618621826]
cls_names[cls]: stop sign

score: 0.9888094
box: [700.6458505483774, -4.298794269561768, 1133.7528921274038, 373.19812774658203]
cls_names[cls]: elephan




bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=frozen_inference_graph.pb
Found 1 possible inputs: (name=image_tensor, type=uint8(4), shape=[?,?,?,3]) 
No variables spotted.
Found 4 possible outputs: (name=detection_boxes, op=Identity) (name=detection_scores, op=Identity) (name=num_detections, op=Identity)
(name=detection_classes, op=Identity) 
Found 6832342 (6.83M) const parameters, 0 (0) variable parameters, and 2089 control_edges
Op types used: 2356 Const, 549 GatherV2, 451 Minimum, 360 Maximum, 287 Reshape, 191 Sub, 183 Cast, 183 Greater, 180 Split, 
180 Where, 119 StridedSlice, 116 Shape, 109 Pack, 101 ConcatV2, 99 Add, 96 Mul, 94 Unpack, 93 Slice, 92 Squeeze, 92 ZerosLike, 
90 NonMaxSuppressionV2, 35 Relu6, 34 BiasAdd, 34 Conv2D, 30 Identity, 29 Switch, 26 Enter, 14 Merge, 13 FusedBatchNorm, 
13 DepthwiseConv2dNative, 13 RealDiv, 12 Range, 11 TensorArrayV3, 8 NextIteration, 8 ExpandDims, 6 TensorArrayWriteV3, 6 TensorArraySizeV3,
6 Exit, 6 TensorArrayGatherV3, 5 TensorArrayReadV3, 5 TensorArrayScatterV3, 4 Fill, 3 Assert, 3 Transpose, 2 LoopCond, 2 Less, 2 Exp, 2 Equal,
1 Size, 1 Sigmoid, 1 ResizeBilinear, 1 Placeholder, 1 Tile, 1 TopKV2
To use with tensorflow/tools/benchmark:benchmark_model try these arguments:
bazel run tensorflow/tools/benchmark:benchmark_model -- --graph=frozen_inference_graph.pb --show_flops --input_layer=image_tensor 
--input_layer_type=uint8 --input_layer_shape=-1,-1,-1,3 --output_layer=detection_boxes,detection_scores,num_detections,detection_classes


ssd_mobile_v1_qu:


tf2onnx.convert: 

Preprocessor/sub', 'FeatureExtractor/MobilenetV1/Conv2d_0/weights/read/_28__cf__31'  frozen_model   frozen_inference_graph

python -m tf2onnx.convert --input frozen_inference_graph.pb --inputs image_tensor:0 --outputs Postprocessor/ExpandDims_1:0,Postprocessor/raw_box_scores:0 --output ssd_mobile_v1_q_decode.onnx

python -m tf2onnx.convert --input frozen_inference_graph.pb --inputs image_tensor:0 --outputs Squeeze:0,concat_1:0 --output ssd_mobile_1.onnx


python -m tf2onnx.convert --input frozen_inference_graph.pb --inputs image_tensor:0 --outputs detection_boxes:0,detection_scores:0,num_detections:0,detection_classes:0 --middle_inputs Preprocessor/sub:0  --middle_outputs Postprocessor/ExpandDims_1:0,Postprocessor/raw_box_scores:0 --output ssd_mobile_v1_q_decode.onnx


----      Postprocessor/ExpandDims_1:0,Postprocessor/Slice:0  Postprocessor/raw_box_scores

python -m tf2onnx.convert --input new_frozen_pb.pb --inputs image_tensor:0 --outputs Squeeze:0,concat_1:0 --output ssd_mobile_v1_new.onnx

run_test:



-------------------------------------

'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/Conv2D', 'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/Conv2D_bn_offset', 'Squeeze', 'concat_1']
            # Preprocessor/map/TensorArrayStack_1/TensorArrayGatherV3:0
            # Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack_1/TensorArrayGatherV3:0
            # Postprocessor/raw_box_encodings:0
            # Postprocessor/scale_logits:0
            # Postprocessor/ToFloat

---------------------------------------------------------

faster_rcnn_resnet50_coco_2018_01_28:




python -m tf2onnx.convert --input frozen_inference_graph.pb --inputs image_tensor:0 --outputs detection_boxes:0,detection_scores:0,num_detections:0,detection_classes:0 --output fasterrcnn.onnx


bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=fastrcnn.pb
Found 1 possible inputs: (name=image_tensor, type=uint8(4), shape=[?,?,?,3]) 
No variables spotted.
Found 4 possible outputs: (name=detection_boxes, op=Identity) (name=detection_scores, op=Identity)
(name=num_detections, op=Identity) (name=detection_classes, op=Identity) 
Found 29166682 (29.17M) const parameters, 0 (0) variable parameters, and 4163 control_edges
Op types used: 4586 Const, 885 StridedSlice, 559 Gather, 485 Mul, 472 Sub, 462 Minimum, 369 Maximum, 
304 Reshape, 276 Split, 205 RealDiv, 204 Pack, 202 ConcatV2, 201 Cast, 188 Greater, 183 Where, 149 Shape, 
128 Add, 106 Slice, 99 Unpack, 97 Squeeze, 94 ZerosLike, 91 NonMaxSuppressionV2, 58 BiasAdd, 56 Conv2D, 55 Enter,
49 Relu, 46 Identity, 45 Switch, 27 Range, 24 Merge, 22 TensorArrayV3, 17 ExpandDims, 15 NextIteration,
12 TensorArrayScatterV3, 12 TensorArrayReadV3, 10 TensorArraySizeV3, 10 TensorArrayWriteV3, 10 Tile, 
10 TensorArrayGatherV3, 10 Exit, 6 Transpose, 6 Fill, 6 Assert, 5 LoopCond, 5 Equal, 5 Less, 4 MaxPool,
4 Exp, 4 Round, 3 Pad, 2 Softmax, 2 Size, 2 TopKV2, 2 GreaterEqual, 2 MatMul, 1 All, 1 CropAndResize,
1 ResizeBilinear, 1 Relu6, 1 Placeholder, 1 LogicalAnd, 1 Mean, 1 Max
To use with tensorflow/tools/benchmark:benchmark_model try these arguments:
bazel run tensorflow/tools/benchmark:benchmark_model -- --graph=fastrcnn.pb --show_flops 
--input_layer=image_tensor --input_layer_type=uint8 --input_layer_shape=-1,-1,-1,3 
--output_layer=detection_boxes,detection_scores,num_detections,detection_classes


ssd-mobile:

WARNING:caffe2.python.workspace:Original python traceback for operator `4` in network `tf2onnx_predict` in exception above (most recent call last):
	run_onnx FAIL [enforce fail at minmax_ops.h:40] output->dims() == Input(i).dims(). 1 149 149 32 vs 1 150 150 32. Description: Input #1, input dimension:1 150 150 32 should match output dimension: 1 149 149 32 Error from operator: 
input: "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNorm:0" input: "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/Relu6__4" output: "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/Relu6:0" name: "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/Relu6" type: "Max" device_option { device_type: 0 cuda_gpu_id: 0 }


caffe2_result: (1, 27816, 4)
caffe2_result: (1, 38, 61, 1024)
	run_onnx OK
diff_result: [[[ 0.0000000e+00 -6.1094761e-07  1.4305115e-06 -7.1525574e-07]
  [ 3.5762787e-07 -2.9802322e-07  1.1920929e-07  4.4703484e-07]
  [ 1.3224781e-07  5.3644180e-07  8.9406967e-08 -4.7683716e-07]
  ...
  [-4.7683716e-07 -3.5762787e-07  7.1525574e-07 -3.5762787e-07]
  [ 1.1920929e-07 -1.7881393e-07 -2.9802322e-08  0.0000000e+00]
  [-1.4901161e-07 -1.5785918e-07 -1.3411045e-07  3.0733645e-08]]]
