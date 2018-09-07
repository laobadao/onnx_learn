import numpy as np
from PIL import Image
import tensorflow as tf
from tools.processor import model_config as config


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def preprocess(resized_inputs):
    """Faster R-CNN Resnet V1 preprocessing.

    VGG style channel mean subtraction as described here:
    https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

    Args:
      resized_inputs: A [batch, height_in, width_in, channels] float32 tensor
        representing a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: A [batch, height_out, width_out, channels] float32
        tensor representing a batch of images.

    """
    channel_means = [123.68, 116.779, 103.939]
    return resized_inputs - [[channel_means]]


def feature_extractor_preprocess(resized_inputs):
    """SSD preprocessing.

    Maps pixel values to the range [-1, 1].

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    return (2.0 / 255.0) * resized_inputs - 1.0


def _compute_new_static_size(image, min_dimension, max_dimension):
    """Compute new static shape for resize_to_range method."""
    image_shape = list(image.shape)
    orig_height = image_shape[0]
    orig_width = image_shape[1]
    num_channels = image_shape[2]
    orig_min_dim = min(orig_height, orig_width)
    # Calculates the larger of the possible sizes
    large_scale_factor = min_dimension / float(orig_min_dim)
    # Scaling orig_(height|width) by large_scale_factor will make the smaller
    # dimension equal to min_dimension, save for floating point rounding errors.
    # For reasonably-sized images, taking the nearest integer will reliably
    # eliminate this error.
    large_height = int(round(orig_height * large_scale_factor))
    large_width = int(round(orig_width * large_scale_factor))
    large_size = [large_height, large_width]
    if max_dimension:
        # Calculates the smaller of the possible sizes, use that if the larger
        # is too big.
        orig_max_dim = max(orig_height, orig_width)
        small_scale_factor = max_dimension / float(orig_max_dim)
        # Scaling orig_(height|width) by small_scale_factor will make the larger
        # dimension equal to max_dimension, save for floating point rounding
        # errors. For reasonably-sized images, taking the nearest integer will
        # reliably eliminate this error.
        small_height = int(round(orig_height * small_scale_factor))
        small_width = int(round(orig_width * small_scale_factor))
        small_size = [small_height, small_width]
        new_size = large_size
        if max(large_size) > max_dimension:
            new_size = small_size
    else:
        new_size = large_size
    return new_size + [num_channels]


def _compute_new_dynamic_size(image, min_dimension, max_dimension):
    """Compute new dynamic shape for resize_to_range method."""
    image_shape = image.size
    orig_height = float(image_shape[1])
    print("orig_height:", orig_height)
    orig_width = float(image_shape[0])
    print("orig_width:", orig_width)
    orig_min_dim = np.minimum(orig_height, orig_width)
    # Calculates the larger of the possible sizes
    min_dimension = float(min_dimension)
    large_scale_factor = min_dimension / orig_min_dim
    # Scaling orig_(height|width) by large_scale_factor will make the smaller
    # dimension equal to min_dimension, save for floating point rounding errors.
    # For reasonably-sized images, taking the nearest integer will reliably
    # eliminate this error.
    large_height = np.int32(np.round(orig_height * large_scale_factor))
    large_width = np.int32(np.round(orig_width * large_scale_factor))
    large_size = np.stack([large_height, large_width])
    if max_dimension:
        # Calculates the smaller of the possible sizes, use that if the larger
        # is too big.
        orig_max_dim = np.maximum(orig_height, orig_width)
        max_dimension = np.float32(max_dimension)
        small_scale_factor = max_dimension / orig_max_dim
        # Scaling orig_(height|width) by small_scale_factor will make the larger
        # dimension equal to max_dimension, save for floating point rounding
        # errors. For reasonably-sized images, taking the nearest integer will
        # reliably eliminate this error.
        small_height = np.int32(np.round(orig_height * small_scale_factor))
        small_width = np.int32(np.round(orig_width * small_scale_factor))
        small_size = np.stack([small_height, small_width])
        new_size = small_size if float(np.max(large_size)) > max_dimension else large_size
    else:
        new_size = large_size
    print("new_size:", new_size)
    return new_size


def preprocessor(img_path, model_type, backend):
    """
    preprocessor image for SSD or faster_rcnn

    Args:
         img_path: a String type value representing image path
         model_type: a String type value representing  "SSD" or "FASTER_RCNN" model
         backend: a String type value "tensorflow" or "numpy" backend

    Returns:
        : resized and Normalized [-1 1] image ndarray
    """

    if not img_path:
        raise ValueError("please input right image path String type")

    if not model_type:
        raise ValueError("model_type must be SSD or FASTER_RCNN String type")

    if not backend:
        raise ValueError("backend must be tensorflow or numpy, String type")

    image = Image.open(img_path)
    if model_type == config.SSD:
        new_size = [300, 300]
    elif model_type == config.FASTER_RCNN:
        min_dim = 600
        max_dim = 1024
        new_size = _compute_new_dynamic_size(image, min_dim, max_dim)
    else:
        raise ValueError("model_type must be SSD or FASTER_RCNN String type")

    if backend == "numpy":
        image = image.resize((new_size[1], new_size[0]), Image.BILINEAR)
        image_np = load_image_into_numpy_array(image)
        print("image_np:", image_np.shape)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        print("image_np_expanded:", image_np_expanded.shape)
        resized_inputs = image_np_expanded.astype(np.float32)

    elif backend == "tensorflow":
        image_np = load_image_into_numpy_array(image)
        image_np_expanded = np.expand_dims(image_np, axis=0)

        with tf.Session() as sess:
            placeholder = tf.placeholder(dtype=tf.uint8, shape=[1, None, None, 3])
            image_np_tf = tf.to_float(placeholder)
            # we only support  ResizeMethod.BILINEAR mode  ResizeBilinear
            resize_image = tf.image.resize_images(image_np_tf, tf.stack([new_size[0], new_size[1]]))
            resized_inputs = sess.run(resize_image, feed_dict={placeholder: image_np_expanded})

    else:
        raise ValueError("backend must be a string value of tensorflow or numpy")

    if model_type == config.SSD:
        resized_inputs = feature_extractor_preprocess(resized_inputs)
    elif model_type == config.FASTER_RCNN:
        # 目前的 预处理只 兼容 resnetv1 ,注：v2 的预处理与 v1 不同，若 要兼容，需要再添加类型的输入
        resized_inputs = preprocess(resized_inputs)

    print("{} {} final_result: {}".format(model_type, backend, resized_inputs.shape))
    return resized_inputs
