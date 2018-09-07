import tensorflow as tf
from tools.processor.utils import anchor_generator_builder, box_list_ops, shape_utils, \
    box_list, post_processing, target_assigner, post_processing_builder, ops
from tools.processor import model_config as config
from tools.processor.utils import standard_fields as fields

BOX_ENCODINGS = 'box_encodings'
CLASS_PREDICTIONS_WITH_BACKGROUND = 'class_predictions_with_background'


# rpn_features_to_crop - FirstStageFeatureExtractor/resnet_v1_50/resnet_v1_50/block3/unit_6/bottleneck_v1/Relu
# rpn_box_predictor_features Conv/Relu6
def crop_and_resize_to_input(rpn_box_predictor_features,
                             preprocessed_inputs, box_encodings,
                             class_predictions_with_background, rpn_features_to_crop):
    image_shape = tf.shape(preprocessed_inputs)

    # rpn_box_encodings, rpn_objectness_predictions_with_background = _predict_rpn_proposals(rpn_box_predictor_features, box_encodings,
    #                             class_predictions_with_background)

    first_stage_anchor_generator = anchor_generator_builder.build("grid_anchor_generator")

    num_anchors_per_location = (
        first_stage_anchor_generator.num_anchors_per_location())

    if len(num_anchors_per_location) != 1:
        raise RuntimeError('anchor_generator is expected to generate anchors '
                           'corresponding to a single feature map.')
    box_predictions = _first_stage_box_predictor_predict([rpn_box_predictor_features], [box_encodings],
                                                         [class_predictions_with_background],
                                                         num_anchors_per_location)

    predictions_box_encodings = tf.concat(
        box_predictions[BOX_ENCODINGS], axis=1)

    print("squeeze predictions_box_encodings.shape:", predictions_box_encodings.shape)

    rpn_box_encodings = tf.squeeze(predictions_box_encodings, axis=2)

    print("rpn_box_encodings.shape:", rpn_box_encodings.shape)

    rpn_objectness_predictions_with_background = tf.concat(
        box_predictions[CLASS_PREDICTIONS_WITH_BACKGROUND],
        axis=1)

    first_stage_anchor_generator = anchor_generator_builder.build("grid_anchor_generator")

    # The Faster R-CNN paper recommends pruning anchors that venture outside
    # the image window at training time and clipping at inference time.
    clip_window = tf.to_float(tf.stack([0, 0, image_shape[1], image_shape[2]]))

    feature_map_shape = tf.shape(rpn_features_to_crop)

    anchors_boxlist = box_list_ops.concatenate(
        first_stage_anchor_generator.generate([(feature_map_shape[1],
                                                feature_map_shape[2])]))

    anchors_boxlist = box_list_ops.clip_to_window(
        anchors_boxlist, clip_window)
    _anchors = anchors_boxlist

    cropped_regions = _predict_second_stage_1(rpn_box_encodings,
                                            rpn_objectness_predictions_with_background,
                                            rpn_features_to_crop,
                                            _anchors.get(),
                                            image_shape)

    return cropped_regions


def _postprocess_rpn(
        rpn_box_encodings_batch,
        rpn_objectness_predictions_with_background_batch,
        anchors,
        image_shapes):
    """Converts first stage prediction tensors from the RPN to proposals.

    This function decodes the raw RPN predictions, runs non-max suppression
    on the result.

    Note that the behavior of this function is slightly modified during
    training --- specifically, we stop the gradient from passing through the
    proposal boxes and we only return a balanced sampled subset of proposals
    with size `second_stage_batch_size`.

    Args:
      rpn_box_encodings_batch: A 3-D float32 tensor of shape
        [batch_size, num_anchors, self._box_coder.code_size] containing
        predicted proposal box encodings.
      rpn_objectness_predictions_with_background_batch: A 3-D float tensor of
        shape [batch_size, num_anchors, 2] containing objectness predictions
        (logits) for each of the anchors with 0 corresponding to background
        and 1 corresponding to object.
      anchors: A 2-D tensor of shape [num_anchors, 4] representing anchors
        for the first stage RPN.  Note that `num_anchors` can differ depending
        on whether the model is created in training or inference mode.
      image_shapes: A 2-D tensor of shape [batch, 3] containing the shapes of
        images in the batch.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.

    Returns:
      proposal_boxes: A float tensor with shape
        [batch_size, max_num_proposals, 4] representing the (potentially zero
        padded) proposal boxes for all images in the batch.  These boxes are
        represented as normalized coordinates.
      proposal_scores:  A float tensor with shape
        [batch_size, max_num_proposals] representing the (potentially zero
        padded) proposal objectness scores for all images in the batch.
      num_proposals: A Tensor of type `int32`. A 1-D tensor of shape [batch]
        representing the number of proposals predicted for each image in
        the batch.
    """
    first_stage_nms_score_threshold = 0.0
    first_stage_nms_iou_threshold = 0.7
    first_stage_max_proposals = 100

    rpn_box_encodings_batch = tf.expand_dims(rpn_box_encodings_batch, axis=2)
    rpn_encodings_shape = shape_utils.combined_static_and_dynamic_shape(
        rpn_box_encodings_batch)
    tiled_anchor_boxes = tf.tile(
        tf.expand_dims(anchors, 0), [rpn_encodings_shape[0], 1, 1])
    print("_batch_decode_boxes 1")
    proposal_boxes = _batch_decode_boxes(rpn_box_encodings_batch,
                                         tiled_anchor_boxes)
    proposal_boxes = tf.squeeze(proposal_boxes, axis=2)
    rpn_objectness_softmax_without_background = tf.nn.softmax(
        rpn_objectness_predictions_with_background_batch)[:, :, 1]
    clip_window = _compute_clip_window(image_shapes)

    (proposal_boxes, proposal_scores, _, _, _,
     num_proposals) = post_processing.batch_multiclass_non_max_suppression(
        tf.expand_dims(proposal_boxes, axis=2),
        tf.expand_dims(rpn_objectness_softmax_without_background,
                       axis=2),
        first_stage_nms_score_threshold,
        first_stage_nms_iou_threshold,
        first_stage_max_proposals,
        first_stage_max_proposals,
        clip_window=clip_window)

    # normalize proposal boxes
    def normalize_boxes(args):
        proposal_boxes_per_image = args[0]
        image_shape = args[1]
        normalized_boxes_per_image = box_list_ops.to_normalized_coordinates(
            box_list.BoxList(proposal_boxes_per_image), image_shape[0],
            image_shape[1], check_range=False).get()
        return normalized_boxes_per_image

    normalized_proposal_boxes = shape_utils.static_or_dynamic_map_fn(
        normalize_boxes, elems=[proposal_boxes, image_shapes], dtype=tf.float32)
    return normalized_proposal_boxes, proposal_scores, num_proposals


def _compute_clip_window(image_shapes):
    """Computes clip window for non max suppression based on image shapes.

    This function assumes that the clip window's left top corner is at (0, 0).

    Args:
      image_shapes: A 2-D int32 tensor of shape [batch_size, 3] containing
      shapes of images in the batch. Each row represents [height, width,
      channels] of an image.

    Returns:
      A 2-D float32 tensor of shape [batch_size, 4] containing the clip window
      for each image in the form [ymin, xmin, ymax, xmax].
    """
    clip_heights = image_shapes[:, 0]
    clip_widths = image_shapes[:, 1]
    clip_window = tf.to_float(tf.stack([tf.zeros_like(clip_heights),
                                        tf.zeros_like(clip_heights),
                                        clip_heights, clip_widths], axis=1))
    return clip_window


def _batch_decode_boxes(box_encodings, anchor_boxes):
    """Decodes box encodings with respect to the anchor boxes.

    Args:
      box_encodings: a 4-D tensor with shape
        [batch_size, num_anchors, num_classes, self._box_coder.code_size]
        representing box encodings.
      anchor_boxes: [batch_size, num_anchors, self._box_coder.code_size]
        representing decoded bounding boxes. If using a shared box across
        classes the shape will instead be
        [total_num_proposals, 1, self._box_coder.code_size].

    Returns:
      decoded_boxes: a
        [batch_size, num_anchors, num_classes, self._box_coder.code_size]
        float tensor representing bounding box predictions (for each image in
        batch, proposal and class). If using a shared box across classes the
        shape will instead be
        [batch_size, num_anchors, 1, self._box_coder.code_size].
    """
    combined_shape = shape_utils.combined_static_and_dynamic_shape(
        box_encodings)
    num_classes = combined_shape[2]

    tiled_anchor_boxes = tf.tile(
        tf.expand_dims(anchor_boxes, 2), [1, 1, num_classes, 1])
    tiled_anchors_boxlist = box_list.BoxList(
        tf.reshape(tiled_anchor_boxes, [-1, 4]))

    _proposal_target_assigner = target_assigner.create_target_assigner(
        'FasterRCNN', 'proposal')
    _box_coder = _proposal_target_assigner.box_coder

    decoded_boxes = _box_coder.decode(
        tf.reshape(box_encodings, [-1, _box_coder.code_size]),
        tiled_anchors_boxlist)

    print("combined_shape[0]:", combined_shape[0])
    print("combined_shape[1]:", combined_shape[1])
    print("num_classes:",num_classes)
    print("decoded_boxes.get():", decoded_boxes.get())

    decoded_boxes_reahpe = tf.reshape(decoded_boxes.get(),
               tf.stack([combined_shape[0], combined_shape[1],
                         num_classes, 4]))

    return decoded_boxes_reahpe


def _image_batch_shape_2d(image_batch_shape_1d):
    """Takes a 1-D image batch shape tensor and converts it to a 2-D tensor.

    Example:
    If 1-D image batch shape tensor is [2, 300, 300, 3]. The corresponding 2-D
    image batch tensor would be [[300, 300, 3], [300, 300, 3]]

    Args:
      image_batch_shape_1d: 1-D tensor of the form [batch_size, height,
        width, channels].

    Returns:
      image_batch_shape_2d: 2-D tensor of shape [batch_size, 3] were each row is
        of the form [height, width, channels].
    """
    return tf.tile(tf.expand_dims(image_batch_shape_1d[1:], 0),
                   [image_batch_shape_1d[0], 1])


def _first_stage_box_predictor_predict(image_features, box_encodings, class_predictions_with_backgrounds,
                                       num_predictions_per_locations):
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
    num_classes = 1
    num_class_slots = num_classes + 1
    _proposal_target_assigner = target_assigner.create_target_assigner(
        'FasterRCNN', 'proposal')
    _box_coder = _proposal_target_assigner.box_coder
    _box_code_size = _box_coder.code_size

    for (image_feature, box_encoding, class_predictions_with_background, num_predictions_per_location) in zip(
            image_features, box_encodings, class_predictions_with_backgrounds, num_predictions_per_locations):
        combined_feature_map_shape = (shape_utils.combined_static_and_dynamic_shape(image_feature))
        print("_box_code_size:", _box_code_size)
        print("num_predictions_per_location:", num_predictions_per_location)
        print("combined_feature_map_shape[1]:", combined_feature_map_shape[1])
        print("combined_feature_map_shape[2]:", combined_feature_map_shape[2])
        print("box_encodings:", box_encoding.shape)

        shapes = tf.stack([combined_feature_map_shape[0],
                           combined_feature_map_shape[1] * combined_feature_map_shape[2] * num_predictions_per_location,
                           1,
                           _box_code_size])

        box_encoding_reshape = tf.reshape(box_encoding, shapes)
        print("box_encoding_reshape:", box_encoding_reshape.shape)

        box_encodings_list.append(box_encoding_reshape)

        class_predictions_with_background = tf.reshape(
            class_predictions_with_background,
            tf.stack([combined_feature_map_shape[0],
                      combined_feature_map_shape[1] *
                      combined_feature_map_shape[2] *
                      num_predictions_per_location,
                      num_class_slots]))

        class_predictions_list.append(class_predictions_with_background)

    return {
        BOX_ENCODINGS: box_encodings_list,
        CLASS_PREDICTIONS_WITH_BACKGROUND: class_predictions_list
    }


def _predict_second_stage_1(rpn_box_encodings,
                          rpn_objectness_predictions_with_background,
                          rpn_features_to_crop,
                          anchors,
                          image_shape):
    image_shape_2d = _image_batch_shape_2d(image_shape)

    proposal_boxes_normalized, _, num_proposals = _postprocess_rpn(
        rpn_box_encodings, rpn_objectness_predictions_with_background,
        anchors, image_shape_2d)

    cropped_regions = (
        _compute_second_stage_input_feature_maps(
            rpn_features_to_crop, proposal_boxes_normalized))

    return cropped_regions


def _compute_second_stage_input_feature_maps(features_to_crop,
                                             proposal_boxes_normalized):
    def get_box_inds(proposals):
        proposals_shape = proposals.get_shape().as_list()
        if any(dim is None for dim in proposals_shape):
            proposals_shape = tf.shape(proposals)
        ones_mat = tf.ones(proposals_shape[:2], dtype=tf.int32)
        multiplier = tf.expand_dims(
            tf.range(start=0, limit=proposals_shape[0]), 1)
        return tf.reshape(ones_mat * multiplier, [-1])

    _initial_crop_size = 14
    cropped_regions = tf.image.crop_and_resize(
        features_to_crop,
        _flatten_first_two_dimensions(proposal_boxes_normalized),
        get_box_inds(proposal_boxes_normalized),
        (_initial_crop_size, _initial_crop_size))

    return cropped_regions


def _flatten_first_two_dimensions(inputs):
    """Flattens `K-d` tensor along batch dimension to be a `(K-1)-d` tensor.

    Converts `inputs` with shape [A, B, ..., depth] into a tensor of shape
    [A * B, ..., depth].

    Args:
      inputs: A float tensor with shape [A, B, ..., depth].  Note that the first
        two and last dimensions must be statically defined.
    Returns:
      A float tensor with shape [A * B, ..., depth] (where the first and last
        dimension are statically defined.
    """
    combined_shape = shape_utils.combined_static_and_dynamic_shape(inputs)
    flattened_shape = tf.stack([combined_shape[0] * combined_shape[1]] +
                               combined_shape[2:])
    return tf.reshape(inputs, flattened_shape)


# rpn_features_to_crop - FirstStageFeatureExtractor/resnet_v1_50/resnet_v1_50/block3/unit_6/bottleneck_v1/Relu
# rpn_box_predictor_features Conv/Relu6

def second_stage_box_predictor(preprocessed_inputs, box_encoding_reshape, class_prediction_reshape,
                               rpn_features_to_crop,
                               rpn_box_encodings,
                               rpn_objectness_predictions_with_background,
                                true_image_shapes,
                               rpn_box_predictor_features):

    image_shape = shape_utils.combined_static_and_dynamic_shape(
        preprocessed_inputs)
    image_shape_2d = _image_batch_shape_2d(image_shape)
    first_stage_anchor_generator = anchor_generator_builder.build("grid_anchor_generator")
    # The Faster R-CNN paper recommends pruning anchors that venture outside
    # the image window at training time and clipping at inference time.
    clip_window = tf.to_float(tf.stack([0, 0, image_shape[1], image_shape[2]]))
    feature_map_shape = tf.shape(rpn_features_to_crop)

    anchors_boxlist = box_list_ops.concatenate(
        first_stage_anchor_generator.generate([(feature_map_shape[1],
                                                feature_map_shape[2])]))
    anchors_boxlist = box_list_ops.clip_to_window(
        anchors_boxlist, clip_window)
    _anchors = anchors_boxlist

    print("second_stage_box_predictor _postprocess_rpn")

    proposal_boxes_normalized, _, num_proposals = _postprocess_rpn(
        rpn_box_encodings, rpn_objectness_predictions_with_background,
        _anchors.get(), image_shape_2d)
    #

    prediction_dict = {
        'rpn_box_predictor_features': rpn_box_predictor_features,
        'rpn_features_to_crop': rpn_features_to_crop,
        'image_shape': image_shape,
        'rpn_box_encodings': rpn_box_encodings,
        'rpn_objectness_predictions_with_background':
            rpn_objectness_predictions_with_background,
    }

    refined_box_encodings = tf.squeeze(
        box_encoding_reshape,
        axis=1, name='all_refined_box_encodings')
    class_predictions_with_background = tf.squeeze(
        class_prediction_reshape,
        axis=1, name='all_class_predictions_with_background')
    _parallel_iterations = 16
    absolute_proposal_boxes = ops.normalized_to_image_coordinates(
        proposal_boxes_normalized, image_shape, _parallel_iterations)

    prediction_dict1 = {
        'refined_box_encodings': refined_box_encodings,
        'class_predictions_with_background':
            class_predictions_with_background,
        'num_proposals': num_proposals,
        'proposal_boxes': absolute_proposal_boxes,
        'box_classifier_features': box_classifier_features,
        'proposal_boxes_normalized': proposal_boxes_normalized,
    }

    prediction_dict.update(prediction_dict1)

    result_output = second_postprocess(prediction_dict, true_image_shapes)

    return result_output

def _predict_second_stage(rpn_box_encodings,
                            rpn_objectness_predictions_with_background,
                            rpn_features_to_crop,
                            anchors,
                            image_shape,
                            true_image_shapes):


    refined_box_encodings = tf.squeeze(
        box_predictions[box_predictor.BOX_ENCODINGS],
        axis=1, name='all_refined_box_encodings')
    class_predictions_with_background = tf.squeeze(
        box_predictions[box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND],
        axis=1, name='all_class_predictions_with_background')

    absolute_proposal_boxes = ops.normalized_to_image_coordinates(
        proposal_boxes_normalized, image_shape, self._parallel_iterations)

    prediction_dict = {
        'refined_box_encodings': refined_box_encodings,
        'class_predictions_with_background':
        class_predictions_with_background,
        'num_proposals': num_proposals,
        'proposal_boxes': absolute_proposal_boxes,
        'box_classifier_features': box_classifier_features,
        'proposal_boxes_normalized': proposal_boxes_normalized,
    }

    return prediction_dict


def second_postprocess(prediction_dict, true_image_shapes):
    """Convert prediction tensors to final detections.

    This function converts raw predictions tensors to final detection results.
    See base class for output format conventions.  Note also that by default,
    scores are to be interpreted as logits, but if a score_converter is used,
    then scores are remapped (and may thus have a different interpretation).

    If number_of_stages=1, the returned results represent proposals from the
    first stage RPN and are padded to have self.max_num_proposals for each
    image; otherwise, the results can be interpreted as multiclass detections
    from the full two-stage model and are padded to self._max_detections.

    Args:
      prediction_dict: a dictionary holding prediction tensors (see the
        documentation for the predict method.  If number_of_stages=1, we
        expect prediction_dict to contain `rpn_box_encodings`,
        `rpn_objectness_predictions_with_background`, `rpn_features_to_crop`,
        and `anchors` fields.  Otherwise we expect prediction_dict to
        additionally contain `refined_box_encodings`,
        `class_predictions_with_background`, `num_proposals`,
        `proposal_boxes` and, optionally, `mask_predictions` fields.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.

    Returns:
      detections: a dictionary containing the following fields
        detection_boxes: [batch, max_detection, 4]
        detection_scores: [batch, max_detections]
        detection_classes: [batch, max_detections]
          (this entry is only created if rpn_mode=False)
        num_detections: [batch]

    Raises:
      ValueError: If `predict` is called before `preprocess`.
    """
    postprocessed_tensors = _postprocess_box_classifier(
        prediction_dict['refined_box_encodings'],
        prediction_dict['class_predictions_with_background'],
        prediction_dict['proposal_boxes'],
        prediction_dict['num_proposals'],
        true_image_shapes,
        mask_predictions=None)

    return _add_output_tensor_nodes(postprocessed_tensors)


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
    if masks is not None:
        tf.add_to_collection(output_collection_name,
                             outputs[detection_fields.detection_masks])
    return outputs


def _postprocess_box_classifier(
        refined_box_encodings,
        class_predictions_with_background,
        proposal_boxes,
        num_proposals,
        image_shapes,
        mask_predictions=None):
    """Converts predictions from the second stage box classifier to detections.

    Args:
      refined_box_encodings: a 3-D float tensor with shape
        [total_num_padded_proposals, num_classes, self._box_coder.code_size]
        representing predicted (final) refined box encodings. If using a shared
        box across classes the shape will instead be
        [total_num_padded_proposals, 1, 4]
      class_predictions_with_background: a 3-D tensor float with shape
        [total_num_padded_proposals, num_classes + 1] containing class
        predictions (logits) for each of the proposals.  Note that this tensor
        *includes* background class predictions (at class index 0).
      proposal_boxes: a 3-D float tensor with shape
        [batch_size, self.max_num_proposals, 4] representing decoded proposal
        bounding boxes in absolute coordinates.
      num_proposals: a 1-D int32 tensor of shape [batch] representing the number
        of proposals predicted for each image in the batch.
      image_shapes: a 2-D int32 tensor containing shapes of input image in the
        batch.
      mask_predictions: (optional) a 4-D float tensor with shape
        [total_num_padded_proposals, num_classes, mask_height, mask_width]
        containing instance mask prediction logits.

    Returns:
      A dictionary containing:
        `detection_boxes`: [batch, max_detection, 4]
        `detection_scores`: [batch, max_detections]
        `detection_classes`: [batch, max_detections]
        `num_detections`: [batch]
        `detection_masks`:
          (optional) [batch, max_detections, mask_height, mask_width]. Note
          that a pixel-wise sigmoid score converter is applied to the detection
          masks.
    """
    _first_stage_max_proposals = 300

    max_num_proposals = _first_stage_max_proposals

    num_classes = 90
    _proposal_target_assigner = target_assigner.create_target_assigner(
        'FasterRCNN', 'proposal')
    _box_coder = _proposal_target_assigner.box_coder
    _second_stage_nms_fn, second_stage_score_conversion_fn = post_processing_builder.build(config.FASTER_RCNN)
    refined_box_encodings_batch = tf.reshape(
        refined_box_encodings,
        [-1,
         max_num_proposals,
         refined_box_encodings.shape[1],
         _box_coder.code_size])
    class_predictions_with_background_batch = tf.reshape(
        class_predictions_with_background,
        [-1, max_num_proposals, num_classes + 1]
    )

    print("_batch_decode_boxes 2")
    refined_decoded_boxes_batch = _batch_decode_boxes(
        refined_box_encodings_batch, proposal_boxes)

    class_predictions_with_background_batch = (
        second_stage_score_conversion_fn(
            class_predictions_with_background_batch))
    class_predictions_batch = tf.reshape(
        tf.slice(class_predictions_with_background_batch,
                 [0, 0, 1], [-1, -1, -1]),
        [-1, max_num_proposals, num_classes])
    clip_window = _compute_clip_window(image_shapes)
    mask_predictions_batch = None
    if mask_predictions is not None:
        mask_height = mask_predictions.shape[2].value
        mask_width = mask_predictions.shape[3].value
        mask_predictions = tf.sigmoid(mask_predictions)
        mask_predictions_batch = tf.reshape(
            mask_predictions, [-1, max_num_proposals,
                               num_classes, mask_height, mask_width])

    (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks, _,
     num_detections) = _second_stage_nms_fn(
        refined_decoded_boxes_batch,
        class_predictions_batch,
        clip_window=clip_window,
        change_coordinate_frame=True,
        num_valid_boxes=num_proposals,
        masks=mask_predictions_batch)
    detections = {
        fields.DetectionResultFields.detection_boxes: nmsed_boxes,
        fields.DetectionResultFields.detection_scores: nmsed_scores,
        fields.DetectionResultFields.detection_classes: nmsed_classes,
        fields.DetectionResultFields.num_detections: tf.to_float(num_detections)
    }
    if nmsed_masks is not None:
        detections[fields.DetectionResultFields.detection_masks] = nmsed_masks
    return detections
