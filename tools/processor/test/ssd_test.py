import tensorflow as tf
import numpy as np
from tensorflow.core.framework import graph_pb2
import tools.processor.postprocessor as post
from tools.processor.utils import visualization_utils as vis_util
from tools.processor.utils import label_map_util
from PIL import Image
from matplotlib import pyplot as plt
import os

IMG_PATH = "E:\\Intenginetech\\tf-onnx\\tools\\processor\\test\\detection_960_660.jpg"
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

np.set_printoptions(threshold=np.inf)

MIDDLE_OUT = None


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def get_beach():
    """Get beach image as input."""

    name = "image_tensor:0"
    image = Image.open(IMG_PATH)
    img_np = load_image_into_numpy_array(image)
    img_np = np.expand_dims(img_np, axis=0)
    return {name: img_np}


def run_tensorflow(sess, inputs, test_name):
    print('run_tensorflow(): so we have a reference output')
    """Run model on tensorflow so we have a referecne output."""
    feed_dict = {}
    for k, v in inputs.items():
        k = sess.graph.get_tensor_by_name(k)
        feed_dict[k] = v

    out_tensor_input = ["Preprocessor/sub:0"]

    out_tensor_middle = ["BoxPredictor_0/BoxEncodingPredictor/BiasAdd:0",
                         "BoxPredictor_1/BoxEncodingPredictor/BiasAdd:0",
                         "BoxPredictor_2/BoxEncodingPredictor/BiasAdd:0",
                         "BoxPredictor_3/BoxEncodingPredictor/BiasAdd:0",
                         "BoxPredictor_4/BoxEncodingPredictor/BiasAdd:0",
                         "BoxPredictor_5/BoxEncodingPredictor/BiasAdd:0",
                         "BoxPredictor_0/ClassPredictor/BiasAdd:0",
                         "BoxPredictor_1/ClassPredictor/BiasAdd:0",
                         "BoxPredictor_2/ClassPredictor/BiasAdd:0",
                         "BoxPredictor_3/ClassPredictor/BiasAdd:0",
                         "BoxPredictor_4/ClassPredictor/BiasAdd:0",
                         "BoxPredictor_5/ClassPredictor/BiasAdd:0",
                         "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6:0",
                         "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6:0",
                         "FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512/Relu6:0",
                         "FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/Relu6:0",
                         "FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_4_3x3_s2_256/Relu6:0",
                         "FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_5_3x3_s2_128/Relu6:0",
                         ]

    out_tensor_final = ["detection_boxes:0",
                        "detection_scores:0",
                        "num_detections:0",
                        "detection_classes:0"]

    test = [test_name]

    test_output = sess.run(test, feed_dict=feed_dict)
    print("test_output shape:", test_output[0].shape)

    global MIDDLE_OUT

    MIDDLE_OUT = test_output[0]

    result_input = sess.run(out_tensor_input, feed_dict=feed_dict)

    result_middle = sess.run(out_tensor_middle, feed_dict=feed_dict)

    result_final = sess.run(out_tensor_final, feed_dict=feed_dict)

    output_dict = {}

    for i in range(len(result_final)):
        print("result_final shape:", result_final[i].shape)

    output_dict["detection_boxes"] = result_final[0]
    output_dict["detection_scores"] = result_final[1]
    output_dict["num_detections"] = result_final[2]
    output_dict["detection_classes"] = result_final[3]

    print("raw tf pb run done")
    return result_input, result_middle, output_dict


def show_detection_result(result, name):
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    # NUM_CLASSES
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    result['num_detections'] = int(result['num_detections'][0])
    result['detection_classes'] = result[
        'detection_classes'][0].astype(np.uint8)
    result['detection_boxes'] = result['detection_boxes'][0]
    result['detection_scores'] = result['detection_scores'][0]

    image = Image.open(IMG_PATH)
    image_np = load_image_into_numpy_array(image)

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        result['detection_boxes'],
        result['detection_classes'],
        result['detection_scores'],
        category_index,
        instance_masks=result.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)

    IMAGE_SIZE = (12, 8)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
    #
    from scipy import misc
    misc.imsave(name+'_detection_result_ssd.png', image_np)
    plt.show()


def main():
    # model_path = "E:\\Intenginetech\\tf-onnx\\tests\\models\\faster_rcnn_resnet50\\frozen_inference_graph.pb"
    model_path = "E:\\Intenginetech\\tf-onnx\\tests\\models\\ssd_mobile\\frozen_inference_graph.pb"
    graph_def = graph_pb2.GraphDef()

    print("model_path:", model_path)
    with open(model_path, "rb") as f:
        graph_def.ParseFromString(f.read())

    g = tf.import_graph_def(graph_def, name='')
    print("g:", g)

    inputs = get_beach()

    test_name = "Postprocessor/BatchMultiClassNonMaxSuppression/mul:0"

    with tf.Session(graph=g) as sess:
        # run the model with tensorflow
        result_input, result_middle, result_final = run_tensorflow(sess, inputs, test_name)

    show_detection_result(result_final, 'raw')

    result, test_output_op = post.run_ssd_tf_post(result_input, result_middle, test_name)
    show_detection_result(result, 'modified')
    check_result(result_final, result, test_output_op)
    print("check_result Done")


def check_result(result_final, result, test_output_op):
    print("detection_scores:", result["detection_scores"])
    print("diff:", result_final["detection_scores"]-result["detection_scores"])
    print("diff one  op out:", test_output_op[0] - MIDDLE_OUT)

if __name__ == "__main__":
    main()
