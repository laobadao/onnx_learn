import tensorflow as tf
from tools.processor.utils import anchor_generator_builder, box_list_ops, shape_utils, post_processing_builder, \
    box_list, box_coder_builder
from tools.processor.utils import standard_fields as fields
import numpy as np
from tools.processor import model_config as config

BOX_ENCODINGS = 'box_encodings'
CLASS_PREDICTIONS_WITH_BACKGROUND = 'class_predictions_with_background'


def post_deal(boxes_encodings, classes_predictions_with_background, feature_maps, preprocessed_inputs,
              true_image_shapes):
    prediction_dict, anchors = last_predict_part(boxes_encodings, classes_predictions_with_background, feature_maps,
                                                 preprocessed_inputs)

    postprocessed_tensors = postprocess(anchors, prediction_dict, true_image_shapes)

    return _add_output_tensor_nodes(postprocessed_tensors)


def run_ssd_tf_post(result_input, result_middle, test_name):
    tf.reset_default_graph()
    print("result_input:", result_input[0].shape)
    print("result_middle:", len(result_middle))
    print("result_final:", )

    input_shape = result_input[0].shape
    result_input_np = result_input[0]
    print("result_input.shape:", input_shape)

    with tf.Session() as sess1:
        result_input_holder = tf.placeholder(dtype=tf.float32,
                                             shape=[input_shape[0], input_shape[1], input_shape[2], input_shape[3]])

        result_input_shape_holder = tf.placeholder(dtype=tf.int32, shape=[1, 3])

        result_input_shape_np = np.array([input_shape[1], input_shape[2], input_shape[3]], dtype=np.int32)
        print("type(result_input_shape_np):", type(result_input_shape_np))
        result_input_shape_np = result_input_shape_np.reshape((1, 3))

        boxes_encodings = []
        classes_predictions_with_background = []
        feature_maps = []

        boxes_encodings_np = []
        classes_predictions_with_background_np = []
        feature_maps_np = []

        for i in range(6):
            shape = result_middle[i].shape
            boxes_encodings.append(tf.placeholder(dtype=tf.float32, shape=[shape[0], shape[1], shape[2], shape[3]]))
            boxes_encodings_np.append(result_middle[i])

        for i1 in range(6, 12, 1):
            shape = result_middle[i1].shape
            classes_predictions_with_background.append(
                tf.placeholder(dtype=tf.float32, shape=[shape[0], shape[1], shape[2], shape[3]]))
            classes_predictions_with_background_np.append(result_middle[i1])

        for i2 in range(12, 18, 1):
            shape = result_middle[i2].shape
            feature_maps.append(
                tf.placeholder(dtype=tf.float32, shape=[shape[0], shape[1], shape[2], shape[3]]))
            feature_maps_np.append(result_middle[i2])

        detections = post_deal(boxes_encodings=boxes_encodings,
                                    classes_predictions_with_background=classes_predictions_with_background,
                                    feature_maps=feature_maps,
                                    preprocessed_inputs=result_input_holder,
                                    true_image_shapes=result_input_shape_holder)
        # feed_dict_1 = {boxes: boxes_np for boxes, boxes_np in zip(boxes_encodings, boxes_encodings_np)}
        # feed_dict_2 = {classes: classes_np for classes, classes_np in
        #                zip(classes_predictions_with_background, classes_predictions_with_background_np)}
        # feed_dict_3 = {feature: feature_np for feature, feature_np in zip(feature_maps, feature_maps_np)}
        #
        #
        # feed_dict[result_input_holder] = result_input_np
        # feed_dict[result_input_shape_holder] = result_input_shape_np

        feed_dict1 = {boxes_encodings[0]: boxes_encodings_np[0],
                      boxes_encodings[1]: boxes_encodings_np[1],
                      boxes_encodings[2]: boxes_encodings_np[2],
                      boxes_encodings[3]: boxes_encodings_np[3],
                      boxes_encodings[4]: boxes_encodings_np[4],
                      boxes_encodings[5]: boxes_encodings_np[5],
                      classes_predictions_with_background[0]:
                          classes_predictions_with_background_np[0],
                      classes_predictions_with_background[1]:
                          classes_predictions_with_background_np[1],
                      classes_predictions_with_background[2]:
                          classes_predictions_with_background_np[2],
                      classes_predictions_with_background[3]:
                          classes_predictions_with_background_np[3],
                      classes_predictions_with_background[4]:
                          classes_predictions_with_background_np[4],
                      classes_predictions_with_background[5]:
                          classes_predictions_with_background_np[5],
                      feature_maps[0]: feature_maps_np[0],
                      feature_maps[1]: feature_maps_np[1],
                      feature_maps[2]: feature_maps_np[2],
                      feature_maps[3]: feature_maps_np[3],
                      feature_maps[4]: feature_maps_np[4],
                      feature_maps[5]: feature_maps_np[5],
                      result_input_holder: result_input_np,
                      result_input_shape_holder: result_input_shape_np
                      }

        test = [test_name]

        test_output_op = sess1.run(test, feed_dict=feed_dict1)
        print("test_output_op shape:", test_output_op[0].shape)


        result = sess1.run(detections, feed_dict=feed_dict1)
        print("result:", len(result))
        print("result detection_boxes:", result["detection_boxes"].shape)
        print("result detection_scores:", result["detection_scores"].shape)
        print("result detection_classes:", result["detection_classes"].shape)
        print("result num_detections:", result["num_detections"].shape)

    return result, test_output_op



def last_predict_part(boxes_encodings, classes_predictions_with_background, feature_maps, preprocessed_inputs):
    """Predicts unpostprocessed tensors from input tensor.

    This function takes an input batch of images and runs it through the forward
    pass of the network to yield unpostprocessesed predictions.

    A side effect of calling the predict method is that self._anchors is
    populated with a box_list.BoxList of anchors.  These anchors must be
    constructed before the postprocess or loss functions can be called.

    Args:
      boxes_encodings:
      classes_predictions_with_background:
      feature_maps:

      preprocessed_inputs: a [batch, height, width, channels] image tensor.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.

    """

    anchor_generator = anchor_generator_builder.build('ssd_anchor_generator')

    prediction_dict = post_processor(boxes_encodings, classes_predictions_with_background,
                                     feature_maps, anchor_generator.num_anchors_per_location())

    image_shape = shape_utils.combined_static_and_dynamic_shape(
        preprocessed_inputs)

    feature_map_spatial_dims = get_feature_map_spatial_dims(
        feature_maps)

    anchors = box_list_ops.concatenate(
        anchor_generator.generate(
            feature_map_spatial_dims,
            im_height=image_shape[1],
            im_width=image_shape[2]))

    box_encodings = tf.concat(prediction_dict['box_encodings'], axis=1)

    print("tf.concat box_encodings:", box_encodings.name)

    if box_encodings.shape.ndims == 4 and box_encodings.shape[2] == 1:
        box_encodings = tf.squeeze(box_encodings, axis=2)

    class_predictions_with_background = tf.concat(
        prediction_dict['class_predictions_with_background'], axis=1)
    predictions_dict = {
        'preprocessed_inputs': preprocessed_inputs,
        'box_encodings': box_encodings,
        'class_predictions_with_background':
            class_predictions_with_background,
        'feature_maps': feature_maps,
        'anchors': anchors.get()
    }
    return predictions_dict, anchors


def get_feature_map_spatial_dims(feature_maps):
    """Return list of spatial dimensions for each feature map in a list.

    Args:
      feature_maps: a list of tensors where the ith tensor has shape
          [batch, height_i, width_i, depth_i].

    Returns:
      a list of pairs (height, width) for each feature map in feature_maps
    """
    feature_map_shapes = [
        shape_utils.combined_static_and_dynamic_shape(
            feature_map) for feature_map in feature_maps
    ]
    return [(shape[1], shape[2]) for shape in feature_map_shapes]


def post_processor(boxes_encodings, classes_predictions_with_background, image_features,
                   num_predictions_per_location_list):
    """Computes encoded object locations and corresponding confidences.

    Args:
      image_features: A list of float tensors of shape [batch_size, height_i,
        width_i, channels_i] containing features for a batch of images.
      num_predictions_per_location_list: A list of integers representing the
        number of box predictions to be made per spatial location for each
        feature map.

    Returns:
      box_encodings: A list of float tensors of shape
        [batch_size, num_anchors_i, q, code_size] representing the location of
        the objects, where q is 1 or the number of classes. Each entry in the
        list corresponds to a feature map in the input `image_features` list.
      class_predictions_with_background: A list of float tensors of shape
        [batch_size, num_anchors_i, num_classes + 1] representing the class
        predictions for the proposals. Each entry in the list corresponds to a
        feature map in the input `image_features` list.
    """

    box_encodings_list = []
    class_predictions_list = []

    for (image_feature,
         num_predictions_per_location,
         box_encodings,
         class_predictions_with_background) in zip(image_features,
                                                   num_predictions_per_location_list,
                                                   boxes_encodings,
                                                   classes_predictions_with_background):
        combined_feature_map_shape = (shape_utils.
                                      combined_static_and_dynamic_shape(image_feature))

        # // Size of the encoding
        # ssd_config.box_predictor.convolutional_box_predictor  box_code_size = 4


        print("num_predictions_per_location:", num_predictions_per_location)
        print("combined_feature_map_shape[1]:", combined_feature_map_shape[1])
        print("combined_feature_map_shape[2]:", combined_feature_map_shape[2])
        print("box_encodings:", box_encodings.shape)

        box_code_size = 4
        box_encodings = tf.reshape(
            box_encodings, tf.stack([combined_feature_map_shape[0],
                                     combined_feature_map_shape[1] *
                                     combined_feature_map_shape[2] *
                                     num_predictions_per_location,
                                     1, box_code_size]))

        print("box_encodings reshape:", box_encodings.name)

        box_encodings_list.append(box_encodings)
        num_classes = 90
        num_class_slots = num_classes + 1

        class_predictions_with_background = tf.reshape(
            class_predictions_with_background,
            tf.stack([combined_feature_map_shape[0],
                      combined_feature_map_shape[1] *
                      combined_feature_map_shape[2] *
                      num_predictions_per_location,
                      num_class_slots]))
        print("class_predictions_with_background reshape:", class_predictions_with_background.name)

        class_predictions_list.append(class_predictions_with_background)

    return {BOX_ENCODINGS: box_encodings_list,
            CLASS_PREDICTIONS_WITH_BACKGROUND: class_predictions_list}


def postprocess(anchors, prediction_dict, true_image_shapes):
    """Converts prediction tensors to final detections.

    This function converts raw predictions tensors to final detection results by
    slicing off the background class, decoding box predictions and applying
    non max suppression and clipping to the image window.

    See base class for output format conventions.  Note also that by default,
    scores are to be interpreted as logits, but if a score_conversion_fn is
    used, then scores are remapped (and may thus have a different
    interpretation).

    Args:
      prediction_dict: a dictionary holding prediction tensors with
        1) preprocessed_inputs: a [batch, height, width, channels] image
          tensor.
        2) box_encodings: 3-D float tensor of shape [batch_size, num_anchors,
          box_code_dimension] containing predicted boxes.
        3) class_predictions_with_background: 3-D float tensor of shape
          [batch_size, num_anchors, num_classes+1] containing class predictions
          (logits) for each of the anchors.  Note that this tensor *includes*
          background class predictions.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros. Or None, if the clip window should cover the full image.

    Returns:
      detections: a dictionary containing the following fields
        detection_boxes: [batch, max_detections, 4]
        detection_scores: [batch, max_detections]
        detection_classes: [batch, max_detections]
        detection_keypoints: [batch, max_detections, num_keypoints, 2] (if
          encoded in the prediction_dict 'box_encodings')
        num_detections: [batch]
    Raises:
      ValueError: if prediction_dict does not contain `box_encodings` or
        `class_predictions_with_background` fields.
    """
    if ('box_encodings' not in prediction_dict or
            'class_predictions_with_background' not in prediction_dict):
        raise ValueError('prediction_dict does not contain expected entries.')
    with tf.name_scope('Postprocessor'):
        preprocessed_images = prediction_dict['preprocessed_inputs']
        box_encodings = prediction_dict['box_encodings']
        box_encodings = tf.identity(box_encodings, 'raw_box_encodings')
        class_predictions = prediction_dict['class_predictions_with_background']
        detection_boxes, detection_keypoints = _batch_decode(anchors, box_encodings)
        detection_boxes = tf.identity(detection_boxes, 'raw_box_locations')
        detection_boxes = tf.expand_dims(detection_boxes, axis=2)
        non_max_suppression_fn, score_conversion_fn = post_processing_builder.build(config.SSD)
        detection_scores_with_background = score_conversion_fn(class_predictions)

        detection_scores_with_background = tf.identity(
            detection_scores_with_background, 'raw_box_scores')
        detection_scores = tf.slice(detection_scores_with_background, [0, 0, 1],
                                    [-1, -1, -1])
        additional_fields = None

        if detection_keypoints is not None:
            additional_fields = {
                fields.BoxListFields.keypoints: detection_keypoints}
        (nmsed_boxes, nmsed_scores, nmsed_classes, _, nmsed_additional_fields,
         num_detections) = non_max_suppression_fn(
            detection_boxes,
            detection_scores,
            clip_window=_compute_clip_window(
                preprocessed_images, true_image_shapes),
            additional_fields=additional_fields)

        detection_dict = {
            fields.DetectionResultFields.detection_boxes: nmsed_boxes,
            fields.DetectionResultFields.detection_scores: nmsed_scores,
            fields.DetectionResultFields.detection_classes: nmsed_classes,
            fields.DetectionResultFields.num_detections:
                tf.to_float(num_detections)
        }
        if (nmsed_additional_fields is not None and
                fields.BoxListFields.keypoints in nmsed_additional_fields):
            detection_dict[fields.DetectionResultFields.detection_keypoints] = (
                nmsed_additional_fields[fields.BoxListFields.keypoints])
        return detection_dict


def _compute_clip_window(preprocessed_images, true_image_shapes):
    """Computes clip window to use during post_processing.

    Computes a new clip window to use during post-processing based on
    `resized_image_shapes` and `true_image_shapes` only if `preprocess` method
    has been called. Otherwise returns a default clip window of [0, 0, 1, 1].

    Args:
      preprocessed_images: the [batch, height, width, channels] image
          tensor.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros. Or None if the clip window should cover the full image.

    Returns:
      a 2-D float32 tensor of the form [batch_size, 4] containing the clip
      window for each image in the batch in normalized coordinates (relative to
      the resized dimensions) where each clip window is of the form [ymin, xmin,
      ymax, xmax] or a default clip window of [0, 0, 1, 1].

    """
    if true_image_shapes is None:
        return tf.constant([0, 0, 1, 1], dtype=tf.float32)

    resized_inputs_shape = shape_utils.combined_static_and_dynamic_shape(
        preprocessed_images)
    true_heights, true_widths, _ = tf.unstack(
        tf.to_float(true_image_shapes), axis=1)
    padded_height = tf.to_float(resized_inputs_shape[1])
    padded_width = tf.to_float(resized_inputs_shape[2])
    return tf.stack(
        [
            tf.zeros_like(true_heights),
            tf.zeros_like(true_widths), true_heights / padded_height,
                                        true_widths / padded_width
        ],
        axis=1)


def _batch_decode(anchors, box_encodings):
    """Decodes a batch of box encodings with respect to the anchors.

    Args:
      box_encodings: A float32 tensor of shape
        [batch_size, num_anchors, box_code_size] containing box encodings.

    Returns:
      decoded_boxes: A float32 tensor of shape
        [batch_size, num_anchors, 4] containing the decoded boxes.
      decoded_keypoints: A float32 tensor of shape
        [batch_size, num_anchors, num_keypoints, 2] containing the decoded
        keypoints if present in the input `box_encodings`, None otherwise.
    """
    combined_shape = shape_utils.combined_static_and_dynamic_shape(
        box_encodings)
    batch_size = combined_shape[0]
    tiled_anchor_boxes = tf.tile(
        tf.expand_dims(anchors.get(), 0), [batch_size, 1, 1])
    tiled_anchors_boxlist = box_list.BoxList(
        tf.reshape(tiled_anchor_boxes, [-1, 4]))

    box_coder = box_coder_builder.build("faster_rcnn_box_coder")

    decoded_boxes = box_coder.decode(
        tf.reshape(box_encodings, [-1, box_coder.code_size]),
        tiled_anchors_boxlist)
    decoded_keypoints = None
    if decoded_boxes.has_field(fields.BoxListFields.keypoints):
        decoded_keypoints = decoded_boxes.get_field(
            fields.BoxListFields.keypoints)
        num_keypoints = decoded_keypoints.get_shape()[1]
        decoded_keypoints = tf.reshape(
            decoded_keypoints,
            tf.stack([combined_shape[0], combined_shape[1], num_keypoints, 2]))
    decoded_boxes = tf.reshape(decoded_boxes.get(), tf.stack(
        [combined_shape[0], combined_shape[1], 4]))
    return decoded_boxes, decoded_keypoints




def _add_output_tensor_nodes(postprocessed_tensors,
                             output_collection_name='inference_op'):
  """Adds output nodes for detection boxes and scores.

  Adds the following nodes for output tensors -
    * num_detections: float32 tensor of shape [batch_size].
    * detection_boxes: float32 tensor of shape [batch_size, num_boxes, 4]
      containing detected boxes.
    * detection_scores: float32 tensor of shape [batch_size, num_boxes]
      containing scores for the detected boxes.
    * detection_classes: float32 tensor of shape [batch_size, num_boxes]
      containing class predictions for the detected boxes.
    * detection_keypoints: (Optional) float32 tensor of shape
      [batch_size, num_boxes, num_keypoints, 2] containing keypoints for each
      detection box.
    * detection_masks: (Optional) float32 tensor of shape
      [batch_size, num_boxes, mask_height, mask_width] containing masks for each
      detection box.

  Args:
    postprocessed_tensors: a dictionary containing the following fields
      'detection_boxes': [batch, max_detections, 4]
      'detection_scores': [batch, max_detections]
      'detection_classes': [batch, max_detections]
      'detection_masks': [batch, max_detections, mask_height, mask_width]
        (optional).
      'num_detections': [batch]
    output_collection_name: Name of collection to add output tensors to.

  Returns:
    A tensor dict containing the added output tensor nodes.
  """
  detection_fields = fields.DetectionResultFields
  label_id_offset = 1
  boxes = postprocessed_tensors.get(detection_fields.detection_boxes)
  scores = postprocessed_tensors.get(detection_fields.detection_scores)
  classes = postprocessed_tensors.get(
      detection_fields.detection_classes) + label_id_offset
  keypoints = postprocessed_tensors.get(detection_fields.detection_keypoints)
  masks = postprocessed_tensors.get(detection_fields.detection_masks)
  num_detections = postprocessed_tensors.get(detection_fields.num_detections)
  outputs = {}
  outputs[detection_fields.detection_boxes] = tf.identity(
      boxes, name=detection_fields.detection_boxes)
  outputs[detection_fields.detection_scores] = tf.identity(
      scores, name=detection_fields.detection_scores)
  outputs[detection_fields.detection_classes] = tf.identity(
      classes, name=detection_fields.detection_classes)
  outputs[detection_fields.num_detections] = tf.identity(
      num_detections, name=detection_fields.num_detections)
  if keypoints is not None:
    outputs[detection_fields.detection_keypoints] = tf.identity(
        keypoints, name=detection_fields.detection_keypoints)
  if masks is not None:
    outputs[detection_fields.detection_masks] = tf.identity(
        masks, name=detection_fields.detection_masks)
  for output_key in outputs:
    tf.add_to_collection(output_collection_name, outputs[output_key])

  return outputs
