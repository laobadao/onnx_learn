import tensorflow as tf
from tensorflow.core.framework import graph_pb2


def run_tensorflow():
    # model_path = "/data1/home/nntool/jjzhao/tf2onnx_Beta1/tf-onnx/tests/models/ssd_mobile/frozen_inference_graph.pb"
    # model_path = "/data1/home/nntool/jjzhao/tf2onnx_Beta1/tf-onnx/tests/models/yolov2_pb/frozen_yolov2.pb"
    # model_path = "/data1/home/nntool/jjzhao/tf2onnx_Beta1/tf-onnx/tests/models/resnet_v1_50/pb_model/frozen_resnet_v1_50.pb"
    # model_path = "/data1/home/nntool/tensorflow/yolov2_optimize_inference_pre.pb"
    # model_path = "faster_optimize_stage2.pb"
    # model_path = "yolov3_optimize_inference_pre.pb"
    model_path = "faster_rcnn_opt_stage1.pb"

    model_path = "yolov2_optimize_inference_pre.pb"

    graph_def = graph_pb2.GraphDef()

    # print("model_path:", model_path)
    with open(model_path, "rb") as f:
        graph_def.ParseFromString(f.read())
    g = tf.import_graph_def(graph_def, name='')

    with tf.Session(graph=g) as sess:
        # run the model with tensorflow

        # input_name = "Preprocessor/sub:0"
        # input_name = "image_tensor:0"
        input_name = "input:0"
        # input_name = "CropAndResize:0"

        inputs = sess.graph.get_tensor_by_name(input_name)
        print("inputs.shape:", inputs.shape)

        shape = [1, 416, 416, 3]
        # shape = [1, 600, 966, 3]
        inputs.set_shape(shape)

        outputs = []
        # outputs_names = ["detection_boxes:0", "detection_scores:0", "num_detections:0", "detection_classes:0"]
        # outputs_names = ["output_bboxes:0", "output_obj:0", "output_class:0"]
        # outputs_names = ["resnet_v1_50/logits/BiasAdd:0"]

        # outputs_names = ["BoxPredictor_0/BoxEncodingPredictor/BiasAdd:0",
        #                  "BoxPredictor_0/ClassPredictor/BiasAdd:0",
        #                  "BoxPredictor_1/BoxEncodingPredictor/BiasAdd:0",
        #                  "BoxPredictor_1/ClassPredictor/BiasAdd:0",
        #                  "BoxPredictor_2/BoxEncodingPredictor/BiasAdd:0",
        #                  "BoxPredictor_2/ClassPredictor/BiasAdd:0",
        #                  "BoxPredictor_3/BoxEncodingPredictor/BiasAdd:0",
        #                  "BoxPredictor_3/ClassPredictor/BiasAdd:0",
        #                  "BoxPredictor_4/BoxEncodingPredictor/BiasAdd:0",
        #                  "BoxPredictor_4/ClassPredictor/BiasAdd:0",
        #                  "BoxPredictor_5/BoxEncodingPredictor/BiasAdd:0",
        #                  "BoxPredictor_5/ClassPredictor/BiasAdd:0", ]

        outputs_names = ["Reshape_1:0"]

        # outputs_names = ["SecondStageBoxPredictor/Reshape:0",
        #                  "SecondStageBoxPredictor/Reshape_1:0"]

        # outputs_names = ["detector/yolo-v3/Reshape_8:0"]

        # outputs_names = ["FirstStageBoxPredictor/BoxEncodingPredictor/BiasAdd:0",
        #                  "FirstStageBoxPredictor/ClassPredictor/BiasAdd:0",
        #                  "Conv/Relu6:0",
        #                  "FirstStageFeatureExtractor/resnet_v1_50/resnet_v1_50/block3/unit_6/bottleneck_v1/Relu:0"]

        for name in outputs_names:
            outputs.append(sess.graph.get_tensor_by_name(name))

        print(outputs)
        tflite_model = tf.contrib.lite.toco_convert(sess.graph_def, [inputs], outputs)
        open("quantized_yolov2.tflite", "wb").write(tflite_model)


if __name__ == "__main__":
    run_tensorflow()
