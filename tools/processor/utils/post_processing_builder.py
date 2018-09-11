# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builder function for post processing operations."""
import functools

import tensorflow as tf
from tools.processor.utils import post_processing
from tools.processor import model_config as config


def build(model_type):
    """Builds callables for post-processing operations.

    Builds callables for non-max suppression and score conversion based on the
    configuration.

    Non-max suppression callable takes `boxes`, `scores`, and optionally
    `clip_window`, `parallel_iterations` `masks, and `scope` as inputs. It returns
    `nms_boxes`, `nms_scores`, `nms_classes` `nms_masks` and `num_detections`. See
    post_processing.batch_multiclass_non_max_suppression for the type and shape
    of these tensors.

    Score converter callable should be called with `input` tensor. The callable
    returns the output from one of 3 tf operations based on the configuration -
    tf.identity, tf.sigmoid or tf.nn.softmax. See tensorflow documentation for
    argument and return value descriptions.

    Args:
      post_processing_config: post_processing.proto object containing the
        parameters for the post-processing operations.

    Returns:
      non_max_suppressor_fn: Callable for non-max suppression.
      score_converter_fn: Callable for score conversion.

    Raises:
      ValueError: if the post_processing_config is of incorrect type.
    """

    if model_type == config.SSD:
        score_converter = "SIGMOID"
    elif model_type == config.FASTER_RCNN:
        score_converter = "SOFTMAX"
    else:
        raise ValueError('type must be ssd or faster_rcnn string')

    non_max_suppressor_fn = _build_non_max_suppressor(model_type)
    # optional float logit_scale = 3[default = 1.0];
    logit_scale = 1.0
    score_converter_fn = _build_score_converter(score_converter, logit_scale)
    return non_max_suppressor_fn, score_converter_fn


def _build_non_max_suppressor(type):
    """Builds non-max suppresson based on the nms config.

    Args:
      nms_config: post_processing_pb2.PostProcessing.BatchNonMaxSuppression proto.

    Returns:
      non_max_suppressor_fn: Callable non-max suppressor.

    Raises:
      ValueError: On incorrect iou_threshold or on incompatible values of
        max_total_detections and max_detections_per_class.
    """

    if type == config.SSD:
        score_threshold = 1e-8
        iou_threshold = 0.6
        max_detections_per_class = 100
        max_total_detections = 100
    elif type == config.FASTER_RCNN:
        score_threshold = 0.0
        iou_threshold = 0.6
        max_detections_per_class = 100
        max_total_detections = 100
    else:
        raise ValueError('type must be ssd or faster_rcnn string')

    if iou_threshold < 0 or iou_threshold > 1.0:
        raise ValueError('iou_threshold not in [0, 1.0].')
    if max_detections_per_class > max_total_detections:
        raise ValueError('max_detections_per_class should be no greater than '
                         'max_total_detections.')

    non_max_suppressor_fn = functools.partial(
        post_processing.batch_multiclass_non_max_suppression,
        score_thresh=score_threshold,
        iou_thresh=iou_threshold,
        max_size_per_class=max_detections_per_class,
        max_total_size=max_total_detections)
    return non_max_suppressor_fn


def _score_converter_fn_with_logit_scale(tf_score_converter_fn, logit_scale):
    """Create a function to scale logits then apply a Tensorflow function."""

    def score_converter_fn(logits):
        scaled_logits = tf.divide(logits, logit_scale, name='scale_logits')

        return tf_score_converter_fn(scaled_logits, name='convert_scores')

    score_converter_fn.__name__ = '%s_with_logit_scale' % (
        tf_score_converter_fn.__name__)
    return score_converter_fn


def _build_score_converter(score_converter, logit_scale):
    """Builds score converter based on the config.

    Builds one of [tf.identity, tf.sigmoid, tf.softmax] score converters based on
    the config.

    Args:
      score_converter_config: post_processing_pb2.PostProcessing.score_converter.
      logit_scale: temperature to use for SOFTMAX score_converter.

    Returns:
      Callable score converter op.

    Raises:
      ValueError: On unknown score converter.
    """

    if score_converter == "SIGMOID":
        return _score_converter_fn_with_logit_scale(tf.sigmoid, logit_scale)
    elif score_converter == "SOFTMAX":
        return _score_converter_fn_with_logit_scale(tf.nn.softmax, logit_scale)

    raise ValueError('Unknown score converter.')
