import numpy as np
from tools.processor import shape_utils
import os
from PIL import Image
from tools.processor import image_resizer_builder

SSD = "SSD"
FASTER_RCNN = "FASTER_RCNN"


# TODO (ZJ): predict 里面最后结尾处一部分 代码 op 需要取出来
# output_tensors = detection_model.predict(
#     preprocessed_inputs, true_image_shapes)
#
#
# TODO: postprocess  也需要重写下
# postprocessed_tensors = detection_model.postprocess(
#     output_tensors, true_image_shapes)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


# "detection_3.jpg"
def build_image_example(img_path):
    """Get ssd faster_rcnn image as inputs and preprocessor."""
    image = Image.open(img_path)
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    return image_np_expanded


def build_graph_preprocessor(img_path, model_config):
    """Build the preprocessor.

    Args:
        img_path: image path
        model_config: SSD or FASTER_RCNN type

    """
    inputs = build_image_example(img_path)
    inputs = inputs.astype(np.float32)
    if model_config == SSD:
        image_resizer_config = "fixed_shape_resizer"
    elif model_config == FASTER_RCNN:
        image_resizer_config = "keep_aspect_ratio_resizer"
    else:
        raise ValueError('model_config must be SSD or FASTER_RCNN')
    image_resizer_fn = image_resizer_builder.build(image_resizer_config)
    preprocessed_inputs = preprocess(inputs, image_resizer_fn)
    return preprocessed_inputs


def preprocess(inputs, image_resizer_fn):
    """Feature-extractor specific preprocessing.

    SSD meta architecture uses a default clip_window of [0, 0, 1, 1] during
    post-processing. On calling `preprocess` method, clip_window gets updated
    based on `true_image_shapes` returned by `image_resizer_fn`.

    Args:
      inputs: a [batch, height_in, width_in, channels] float tensor representing
        a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: a [batch, height_out, width_out, channels] float
        tensor representing a batch of images.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.

    Raises:
      ValueError: if inputs tensor does not have type tf.float32
    """
    if inputs.dtype is not np.float32:
        raise ValueError('`preprocess` expects a tf.float32 tensor')

    # TODO(jonathanhuang): revisit whether to always use batch size as
    # the number of parallel iterations vs allow for dynamic batching.
    outputs = shape_utils.static_or_dynamic_map_fn(
        image_resizer_fn,
        elems=inputs,
        dtype=[np.float32, np.int32])
    resized_inputs = outputs[0]
    true_image_shapes = outputs[1]

    return feature_extractor_preprocess(resized_inputs), true_image_shapes


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


def rgb_to_gray(image):
    """Converts a 3 channel RGB image to a 1 channel grayscale image.

    Args:
      image: Rank 3 float32 tensor containing 1 image -> [height, width, 3]
             with pixel values varying between [0, 1].

    Returns:
      image: A single channel grayscale image -> [image, height, 1].
    """
    return _rgb_to_grayscale(image)


def _rgb_to_grayscale(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


# TODO(alirezafathi): Investigate if instead the function should return None if
# masks is None.
# pylint: disable=g-doc-return-or-yield
def resize_image(image,
                 new_height=600,
                 new_width=1024,
                 method=Image.BILINEAR):
    """Resizes images to the given height and width.

    Args:
      image: A 3D tensor of shape [height, width, channels]
      masks: (optional) rank 3 float32 tensor with shape
             [num_instances, height, width] containing instance masks.
      new_height: (optional) (scalar) desired height of the image.
      new_width: (optional) (scalar) desired width of the image.
      method: (optional) interpolation method used in resizing. Defaults to
              BILINEAR.
      align_corners: bool. If true, exactly align all 4 corners of the input
                     and output. Defaults to False.

    Returns:
      Note that the position of the resized_image_shape changes based on whether
      masks are present.
      resized_image: A tensor of size [new_height, new_width, channels].
      resized_masks: If masks is not None, also outputs masks. A 3D tensor of
        shape [num_instances, new_height, new_width]
      resized_image_shape: A 1D tensor of shape [3] containing the shape of the
        resized image.
    """

    new_image = image.resize(image, method)
    image_shape = shape_utils.combined_static_and_dynamic_shape(image)
    result = new_image
    result.append(np.stack([new_height, new_width, image_shape[2]]))
    return result


def resize_to_range(image,
                    min_dimension=600,
                    max_dimension=1024,
                    method=Image.BILINEAR,
                    pad_to_max_dimension=False,
                    per_channel_pad_value=(0, 0, 0)):
    """Resizes an image so its dimensions are within the provided value.

    The output size can be described by two cases:
    1. If the image can be rescaled so its minimum dimension is equal to the
       provided value without the other dimension exceeding max_dimension,
       then do so.
    2. Otherwise, resize so the largest dimension is equal to max_dimension.

    Args:
      image: A 3D tensor of shape [height, width, channels]
      masks: (optional) rank 3 float32 tensor with shape
             [num_instances, height, width] containing instance masks.
      min_dimension: (optional) (scalar) desired size of the smaller image
                     dimension.
      max_dimension: (optional) (scalar) maximum allowed size
                     of the larger image dimension.
      method: (optional) interpolation method used in resizing. Defaults to
              BILINEAR.


    Returns:
      Note that the position of the resized_image_shape changes based on whether
      masks are present.
      resized_image: A 3D tensor of shape [new_height, new_width, channels],
        where the image has been resized (with bilinear interpolation) so that
        min(new_height, new_width) == min_dimension or
        max(new_height, new_width) == max_dimension.
      resized_masks: If masks is not None, also outputs masks. A 3D tensor of
        shape [num_instances, new_height, new_width].
      resized_image_shape: A 1D tensor of shape [3] containing shape of the
        resized image.

    Raises:
      ValueError: if the image is not a 3D tensor.
    """
    if len(image.shape) != 3:
        raise ValueError('Image should be 3D tensor')

    if image.get_shape().is_fully_defined():
        new_size = _compute_new_static_size(image, min_dimension, max_dimension)
    else:
        new_size = _compute_new_dynamic_size(image, min_dimension, max_dimension)
    new_image = image.resize(new_size[:-1], method)

    if pad_to_max_dimension:
        channels = np.split(new_image, len(new_image), axis=2)
        if len(channels) != len(per_channel_pad_value):
            raise ValueError('Number of channels must be equal to the length of '
                             'per-channel pad value.')
        new_image = np.stack(
            [
                np.pad(
                    channels[i], [[0, max_dimension - new_size[0]],
                                  [0, max_dimension - new_size[1]]],
                    constant_values=per_channel_pad_value[i])
                for i in range(len(channels))
            ],
            axis=2)
        new_image.reshape((max_dimension, max_dimension, 3))

    result = new_image
    result.append(new_size)
    return result


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
    image_shape = image.shape
    orig_height = float(image_shape[0])
    orig_width = float(image_shape[1])
    num_channels = image_shape[2]
    orig_min_dim = np.minimum(orig_height, orig_width)
    # Calculates the larger of the possible sizes
    min_dimension = np.float32(min_dimension)
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
    return np.stack(np.split(new_size, len(new_size)) + [num_channels])
