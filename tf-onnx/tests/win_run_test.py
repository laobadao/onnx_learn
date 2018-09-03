import tensorflow as tf
import numpy as np
from tensorflow.core.framework import graph_pb2
import tf2onnx


def get_beach():
    """Get beach image as input."""
    img_np = np.ones(shape=(1, 299, 299, 3), dtype=np.float32)
    print("ones img_np:", img_np.shape)

    # name = "detector/truediv:0"
    name = "input:0"
    # name = "image_tensor:0"

    return {name: img_np}


def run_tensorflow(sess, inputs):
    print('run_tensorflow(): so we have a reference output')
    """Run model on tensorflow so we have a referecne output."""
    feed_dict = {}
    for k, v in inputs.items():
        k = sess.graph.get_tensor_by_name(k)
        feed_dict[k] = v

    # out_tensor = ["resnet_v1_50/logits/BiasAdd:0"]

    # out_tensor = ["vgg_16/fc8/squeezed:0"]
    # out_tensor = ["conv_dec/BiasAdd:0"]
    # out_tensor = ["Reshape_1:0"]
    # out_tensor = ["detector/yolo-v3/Reshape_8:0"]
    # out_tensor = ["InceptionResnetV2/Logits/Predictions:0"]

    # out_tensor = ["FirstStageBoxPredictor/BoxEncodingPredictor/BiasAdd:0",
    #               "FirstStageBoxPredictor/ClassPredictor/BiasAdd:0",
    #               "Conv/Relu6:0",
    #               "FirstStageFeatureExtractor/resnet_v1_50/resnet_v1_50/block3/unit_6/bottleneck_v1/Relu:0"]

    # out_tensor = ["InceptionV4/Logits/Predictions:0"]

    # out_tensor = ["MobilenetV1/Predictions/Softmax:0"]

    # out_tensor = ["InceptionV1/Logits/Predictions/Softmax:0"]

    out_tensor = ["InceptionV3/Predictions/Softmax:0"]

    result = sess.run(out_tensor, feed_dict=feed_dict)

    # print("result:", result[0].shape)

    print("len(result):", len(result))
    for i in range(len(result)):
        print("result:", result[i].shape)


    import h5py
    f = h5py.File("InceptionV3.h5", "w")
    f.create_dataset("InceptionV3", data=result[0][0])
    f.close()

    # import h5py
    # f = h5py.File("faster_rcnn_resnet50.h5", "w")
    # for i in range(len(result)):
    #     f["result"+ str(i)]=result[i][0]
    # f.close()

    # if result[0].shape == (1, 1, 1, 1001) or result[0].shape == (1, 1, 1, 1000):
    #     print(" result.shape:", result[0][0][0].shape)
    #     index = np.argmax(result[0][0][0], axis=1)
    #     print("\t result[0][0][0][index]:", result[0][0][0][0][index])
    #     print("\t index:", index)
    # if result[0].shape == (1, 1000):
    #     print(" result.shape:", result[0].shape)
    #     index = np.argmax(result[0], axis=1)
    #     print("\t result[0][0][0][index]:", result[0][0][index])
    #     print("\t index:", index)

    return result


def main():
    # model_path = "E:\\Intenginetech\\tf-onnx-normal\\tests\\models\\vgg16\\frozen_model_vgg_16.pb"

    # model_path = "E:\\Intenginetech\\tf-onnx\\tests\\models\\yolov2_pb\\frozen_yolov2_space.pb"
    # model_path = "E:\\Intenginetech\\tf-onnx\\tests\\models\\yolov2_pb\\frozen_yolov2.pb"
    # model_path = "E:\\Intenginetech\\tf-onnx\\tests\\models\\yolov3\\frozen_yolov3.pb"

    # model_path = "E:\\Intenginetech\\tf-onnx\\tests\\models\\inception_resnet_v2\\inception_resnet_v2_2016_08_30_frozen.pb"

    # model_path = "E:\\Intenginetech\\tf-onnx\\tests\\models\\faster_rcnn_resnet50\\frozen_inference_graph.pb"
    # model_path = "E:\\Intenginetech\\tf-onnx\\tests\\models\\inceptionV4\\inception_v4_2016_09_09_frozen.pb"

    # model_path = "E:\\Intenginetech\\tf-onnx\\tests\\models\\mobilenetV1_224\\mobilenet_v1_1.0_224\\frozen_graph.pb"

    model_path = "E:\\Intenginetech\\tf-onnx\\tests\\models\\inception_v3\\inception_v3_2016_08_28_frozen.pb"

    graph_def = graph_pb2.GraphDef()
    print("model_path:", model_path)
    with open(model_path, "rb") as f:
        graph_def.ParseFromString(f.read())
    g = tf.import_graph_def(graph_def, name='')
    print("g:", g)

    inputs = get_beach()

    with tf.Session(graph=g) as sess:
        # run the model with tensorflow
        tf_results = run_tensorflow(sess, inputs)
        print("done")


if __name__ == "__main__":
    main()
