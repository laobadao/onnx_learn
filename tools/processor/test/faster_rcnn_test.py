import tensorflow as tf
import numpy as np
from tensorflow.core.framework import graph_pb2
import tools.processor.faster_rcnn_builder as frcnn
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

    out_tensor_middle = ["FirstStageBoxPredictor/BoxEncodingPredictor/BiasAdd:0",
                         "FirstStageBoxPredictor/ClassPredictor/BiasAdd:0",
                         "Conv/Relu6:0",
                         "FirstStageFeatureExtractor/resnet_v1_50/resnet_v1_50/block3/unit_6/bottleneck_v1/Relu:0",
                         "SecondStageBoxPredictor/Reshape:0",
                         "SecondStageBoxPredictor/Reshape_1:0"
                         ]

    out_tensor_final = ["detection_boxes:0",
                        "detection_scores:0",
                        "num_detections:0",
                        "detection_classes:0"]

    crop_and_resize_tensor = ["CropAndResize:0"]

    test = [test_name]

    test_output = sess.run(test, feed_dict=feed_dict)
    print("test_output shape:", test_output[0].shape)

    global MIDDLE_OUT

    MIDDLE_OUT = test_output[0]
    result_input = sess.run(out_tensor_input, feed_dict=feed_dict)
    result_middle = sess.run(out_tensor_middle, feed_dict=feed_dict)
    crop_and_resize_out = sess.run(crop_and_resize_tensor, feed_dict=feed_dict)
    print("crop_and_resize_out shape:", crop_and_resize_out[0].shape)

    result_final = sess.run(out_tensor_final, feed_dict=feed_dict)
    #
    output_dict = {}
    #
    for i in range(len(result_final)):
        print("result_final shape:", result_final[i].shape)
    #
    output_dict["detection_boxes"] = result_final[0]
    output_dict["detection_scores"] = result_final[1]
    output_dict["num_detections"] = result_final[2]
    output_dict["detection_classes"] = result_final[3]

    print("raw tf pb run done")
    return result_input, result_middle, crop_and_resize_out, output_dict


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
    misc.imsave(name + '_detection_result.png', image_np)
    plt.show()


def main():
    model_path = "E:\\Intenginetech\\tf-onnx\\tests\\models\\faster_rcnn_resnet50\\frozen_inference_graph.pb"
    graph_def = graph_pb2.GraphDef()

    print("model_path:", model_path)
    with open(model_path, "rb") as f:
        graph_def.ParseFromString(f.read())

    g = tf.import_graph_def(graph_def, name='')
    print("g:", g)

    inputs = get_beach()

    test_name = "CropAndResize:0"

    with tf.Session(graph=g) as sess:
        # run the model with tensorflow
        result_input, result_middle, crop_and_resize_out, output_dict = run_tensorflow(sess, inputs, test_name)

    show_detection_result(output_dict, 'raw')
    #
    result, test_output_op = run_faster_rcnn_tf_post(result_input, result_middle, crop_and_resize_out, test_name)
    show_detection_result(result, 'modified')

    check_result(output_dict, result, test_output_op)

    print("check_result Done")


def run_faster_rcnn_tf_post(result_input, result_middle, crop_and_resize_out, test_name):
    tf.reset_default_graph()
    input_shape = result_input[0].shape
    print("result_input.shape:", input_shape)
    preprocessed_inputs_np = result_input[0]
    box_encodings = result_middle[0]
    class_predictions_with_background = result_middle[1]

    rpn_box_predictor_features = result_middle[2]
    print("rpn_box_predictor_features.shape:", rpn_box_predictor_features.shape)
    rpn_features_to_crop = result_middle[3]

    with tf.Session() as sess1:
        result_input_holder = tf.placeholder(dtype=tf.float32,
                                             shape=[input_shape[0], input_shape[1], input_shape[2], input_shape[3]])

        rpn_box_predictor_features_holder = tf.placeholder(dtype=tf.float32, shape=[rpn_box_predictor_features.shape[0],
                                                                                    rpn_box_predictor_features.shape[1],
                                                                                    rpn_box_predictor_features.shape[2],
                                                                                    rpn_box_predictor_features.shape[3]])

        box_encodings_holder = tf.placeholder(dtype=tf.float32, shape=[box_encodings.shape[0],
                                                                       box_encodings.shape[1],
                                                                       box_encodings.shape[2],
                                                                       box_encodings.shape[3]])
        class_predictions_with_background_holder = tf.placeholder(dtype=tf.float32,
                                                                  shape=[class_predictions_with_background.shape[0],
                                                                         class_predictions_with_background.shape[1],
                                                                         class_predictions_with_background.shape[2],
                                                                         class_predictions_with_background.shape[3]])

        rpn_features_to_crop_holder = tf.placeholder(dtype=tf.float32, shape=[rpn_features_to_crop.shape[0],
                                                                              rpn_features_to_crop.shape[1],
                                                                              rpn_features_to_crop.shape[2],
                                                                              rpn_features_to_crop.shape[3]])

        cropped_regions = frcnn.crop_and_resize_to_input(rpn_box_predictor_features=rpn_box_predictor_features_holder,
                                                         preprocessed_inputs=result_input_holder,
                                                         box_encodings=box_encodings_holder,
                                                         class_predictions_with_background=class_predictions_with_background_holder,
                                                         rpn_features_to_crop=rpn_features_to_crop_holder)

        print("cropped_regions:", cropped_regions.shape)

        feed_dict1 = {result_input_holder: preprocessed_inputs_np,
                      rpn_box_predictor_features_holder: rpn_box_predictor_features,
                      box_encodings_holder: box_encodings,
                      class_predictions_with_background_holder: class_predictions_with_background,
                      rpn_features_to_crop_holder: rpn_features_to_crop}

        # test = [test_name]
        test_output_op = None
        # test_output_op = sess1.run(test, feed_dict=feed_dict1)

        # print("test_output_op shape:", test_output_op[0].shape)

        result = sess1.run(cropped_regions, feed_dict=feed_dict1)

        box_encoding_reshape = result_middle[4]
        class_prediction_reshape = result_middle[5]

        print("box_encoding_reshape:", box_encoding_reshape.shape)
        print("class_prediction_reshape:", class_prediction_reshape.shape)

        box_encoding_reshape_holder = tf.placeholder(dtype=tf.float32, shape=[box_encoding_reshape.shape[0],
                                                                              box_encoding_reshape.shape[1],
                                                                              box_encoding_reshape.shape[2],
                                                                              box_encoding_reshape.shape[3]])
        class_prediction_reshape_holder = tf.placeholder(dtype=tf.float32, shape=[class_prediction_reshape.shape[0],
                                                                                  class_prediction_reshape.shape[1],
                                                                                  class_prediction_reshape.shape[2]
                                                                                  ])

        result_input_shape_np = np.array([input_shape[1], input_shape[2], input_shape[3]], dtype=np.int32)
        print("type(result_input_shape_np):", type(result_input_shape_np))
        result_input_shape_np = result_input_shape_np.reshape((1, 3))
        result_input_shape_holder = tf.placeholder(dtype=tf.int32, shape=[1, 3])

        second_stage_out = frcnn.second_stage_box_predictor(
                                preprocessed_inputs=result_input_holder,
                                box_encoding_reshape=box_encoding_reshape_holder,
                                class_prediction_reshape=class_prediction_reshape_holder,
                                rpn_features_to_crop=rpn_features_to_crop_holder,
                                rpn_box_encodings=box_encodings_holder,
                                rpn_objectness_predictions_with_background=class_predictions_with_background_holder,
                                true_image_shapes=result_input_shape_holder,
                                rpn_box_predictor_features=rpn_box_predictor_features_holder
        )

        feed_dict1[box_encoding_reshape_holder] = box_encoding_reshape
        feed_dict1[class_prediction_reshape_holder] = class_prediction_reshape
        feed_dict1[result_input_shape_holder] = result_input_shape_np

        result2 = sess1.run(second_stage_out, feed_dict=feed_dict1)

        # print("result2:", result2)

        # print("crop_and_resize_out[0]:", crop_and_resize_out[0][0].shape)
        # print("diff crop_and_resize:", crop_and_resize_out[0][0] - result[0])

        print("result detection_boxes:", result2["detection_boxes"].shape)
        print("result detection_scores:", result2["detection_scores"].shape)
        print("result detection_classes:", result2["detection_classes"].shape)
        print("result num_detections:", result2["num_detections"].shape)

    return result2, test_output_op


def check_result(result_final, result, test_output_op):
    print("detection_scores:", result["detection_scores"].shape)
    print("detection_scores:", result_final["detection_scores"].shape)
    print("diff:", result_final["detection_scores"] - result["detection_scores"])
    # print("diff one  op out:", test_output_op[0] - MIDDLE_OUT)


if __name__ == "__main__":
    main()
