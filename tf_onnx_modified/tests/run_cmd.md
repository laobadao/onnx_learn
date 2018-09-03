ssd_mobile_v1_qu:  OK

tf2onnx:

python -m tf2onnx.convert --input frozen_inference_graph.pb --inputs image_tensor:0[1,300,300,3] --outputs detection_boxes:0,detection_scores:0,num_detections:0,detection_classes:0 --middle_inputs Preprocessor/sub:0 --middle_outputs BoxPredictor_0/BoxEncodingPredictor/BiasAdd:0,BoxPredictor_0/ClassPredictor/BiasAdd:0,BoxPredictor_1/BoxEncodingPredictor/BiasAdd:0,BoxPredictor_1/ClassPredictor/BiasAdd:0,BoxPredictor_2/BoxEncodingPredictor/BiasAdd:0,BoxPredictor_2/ClassPredictor/BiasAdd:0,BoxPredictor_3/BoxEncodingPredictor/BiasAdd:0,BoxPredictor_4/BoxEncodingPredictor/BiasAdd:0,BoxPredictor_4/ClassPredictor/BiasAdd:0,BoxPredictor_5/BoxEncodingPredictor/BiasAdd:0,BoxPredictor_5/ClassPredictor/BiasAdd:0 --output ssd_mobile_v1_q.onnx

run_test:

参数配置内容查看 ssd_mobile.yaml

python3 tests/convert_models.py --config tests/ssd_mobile.yaml  --onnx-file ssd_mobile


------

VGG16:  OK


tf2onnx:

参数配置内容查看 vgg16.yaml

python3 -m tf2onnx.convert --input tests/models/vgg16/frozen_model_vgg_16.pb --inputs input:0  --outputs vgg_16/fc8/squeezed:0 --output vgg_16_squeeze.onnx

run_test:

python3 tests/convert_models.py --config tests/vgg16.yaml  --onnx-file vgg16

-----

resnet_v1_50:  OK


python3 tests/convert_models.py --config tests/resnet_v1_50.yaml  --onnx-file resnet_v1_50

python3 -m tf2onnx.convert --input tests/models/resnet_v1_50/pb_model/frozen_resnet_v1_50.pb --inputs input:0 --outputs resnet_v1_50/logits/BiasAdd:0 --output resnet_v1_50.onnx

----

resnet_v2_50:  OK


python3 tests/convert_models.py --config tests/resnet_v2_50.yaml  --onnx-file resnet_v2_50


python3 -m tf2onnx.convert --input tests/models/resnet_v2_50/frozen_resnet_v2_50.pb --inputs input:0 --outputs resnet_v2_50/logits/BiasAdd:0 --output resnet_v2_50.onnx

----

----

yolov2: spacetodepth

python3 tests/convert_models.py --config tests/yolov2_space.yaml --onnx-file yolov2_space

python3 -m tf2onnx.convert --input tests/models/yolov2_pb/frozen_yolov2_space.pb --inputs input:0 --outputs conv_dec/BiasAdd:0 --output yolov2_space.onnx



----

inceptionv3:


python3 tests/convert_models.py --config tests/inception_v3_slim.yaml --onnx-file inception_v3_slim


python3 -m tf2onnx.convert --input tests/models/inception_v3/inception_v3_2016_08_28_frozen.pb --inputs input:0 --outputs InceptionV3/Predictions/Softmax:0 --output inception_v3.onnx

-----------

inception_v1:


python3 tests/convert_models.py --config tests/googlenet_v1_slim.yaml --onnx-file googlenet_v1_slim


python3 -m tf2onnx.convert --input tests/models/inceptionV1/inception_v1_2016_08_28_frozen.pb --inputs input:0 --outputs InceptionV1/Logits/Predictions/Softmax:0 --output InceptionV1.onnx

-----------

inception_v2:

python3 tests/convert_models.py --config tests/googlenet_resnet_v2.yaml --onnx-file googlenet_resnet_v2


python3 -m tf2onnx.convert --input tests/models/inception_resnet_v2/inception_resnet_v2_2016_08_30_frozen.pb --inputs input:0 --outputs InceptionResnetV2/Logits/Predictions:0 --output InceptionResnetV2.onnx

-----------


inception_v4:

python3 tests/convert_models.py --config tests/googlenet_v4_slim.yaml --onnx-file googlenet_v4_slim

python3 -m tf2onnx.convert --input tests/models/inceptionV4/inception_v4_2016_09_09_frozen.pb --inputs input:0 --outputs InceptionV4/Logits/Predictions:0 --output InceptionV4.onnx

-----------
mobilenet_v1_75_192

python3 tests/convert_models.py --config tests/mobilenet_v1_75_192.yaml --onnx-file mobilenet_v1_75_192

python3 -m tf2onnx.convert --input tests/models/mobilenetV1_224/mobilenet_v1_1.0_224/frozen_graph.pb --inputs input:0 --outputs MobilenetV1/Predictions/Softmax:0 --output MobilenetV1_224.onnx

-----------
mobilenet_v1_100_224

python3 tests/convert_models.py --config tests/mobilenet_v1_100_224.yaml --onnx-file mobilenet_v1_100_224


python3 -m tf2onnx.convert --input tests/models/mobilenetV1_192/mobilenet_v1_0.75_192/frozen_graph.pb --inputs input:0 --outputs MobilenetV1/Predictions/Softmax:0 --output MobilenetV1_224.onnx

-----------

faster_rcnn_resnet50：


python3 tests/convert_models.py --config tests/faster_rcnn_resnet50.yaml --onnx-file faster_rcnn_resnet50


python -m tf2onnx.convert --input tests/models/faster_rcnn_resnet50/frozen_inference_graph.pb --inputs image_tensor:0[1,600,966,3] --outputs detection_boxes:0,detection_scores:0,num_detections:0,detection_classes:0 --middle_inputs Preprocessor/sub:0 --middle_outputs FirstStageBoxPredictor/BoxEncodingPredictor/BiasAdd:0,FirstStageBoxPredictor/ClassPredictor/BiasAdd:0,Conv/Relu6:0,FirstStageFeatureExtractor/resnet_v1_50/resnet_v1_50/block3/unit_6/bottleneck_v1/Relu:0 --output faster_rcnn_stage1.onnx


python3 tests/convert_models.py --config tests/faster_rcnn_resnet50_2.yaml --onnx-file faster_rcnn_resnet50_2

# TODO 这里还不 OK

multiple_parts: True

python -m tf2onnx.convert --input tests/models/faster_rcnn_resnet50/frozen_inference_graph.pb --inputs image_tensor:0[100, 14, 14, 1024] --outputs detection_boxes:0,detection_scores:0,num_detections:0,detection_classes:0 --middle_inputs CropAndResize:0 --middle_outputs SecondStageBoxPredictor/Reshape:0,SecondStageBoxPredictor/Reshape_1:0 --output faster_rcnn_stage2.onnx
-----

yolov3:

python3 tests/convert_models.py --config tests/yolov3.yaml --onnx-file yolov3


python3 -m tf2onnx.convert --input tests/models/yolov3/frozen_yolov3.pb --inputs input:0 --outputs output:0 --middle_inputs detector/truediv:0 --middle_outputs detector/yolo-v3/Reshape_8:0 --output yolov3.onnx

----------

--pdf-engine=xelatex



pandoc -N -s --pdf-engine=xelatex --template=E:\\software\\intengine_doc_template.tex IR.md -o  IR.pdf


pandoc -N -s  README.md --pdf-engine=xelatex --template=/data1/home/nntool/jjzhao/software/intengine_doc_template.tex  -o  tf2onnx2Keras.pdf


pandoc -N -s  Operators.md --latex-engine=xelatex --template=/data1/home/nntool/jjzhao/software/intengine_doc_template.tex  -o  Operators.pdf

gen_pdf README.md tf2onnx2Keas.pdf

gen_pdf README.md tf2onnx2Keas.pdf