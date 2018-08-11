"""Builder function for image resizing operations."""
import functools
from tools.processor import preprocessor
import numpy as np
from PIL import Image


def build(image_resizer_config):
    """Builds callable for image resizing operations.

    Args:
      image_resizer_config: image_resizer.proto object containing parameters for
        an image resizing operation.

    Returns:
      image_resizer_fn: Callable for image resizing.  This callable always takes
        a rank-3 image tensor (corresponding to a single image) and returns a
        rank-3 image tensor, possibly with new spatial dimensions.

    Raises:
      ValueError: if `image_resizer_config` is of incorrect type.
      ValueError: if `image_resizer_config.image_resizer_oneof` is of expected
        type.
      ValueError: if min_dimension > max_dimension when keep_aspect_ratio_resizer
        is used.
    """
    if image_resizer_config == 'keep_aspect_ratio_resizer':
        # TODO ssd faster rcnn configs 中 固定的 参数配置，在这里拆解 交叉对比下 关于 预处理部分的参数配置 哪些一样 哪些不一样
        min_dimension = 600
        max_dimension = 1024
        method = Image.BILINEAR
        image_resizer_fn = functools.partial(
            preprocessor.resize_to_range,
            min_dimension=min_dimension,
            max_dimension=max_dimension,
            method=method)
    elif image_resizer_config == 'fixed_shape_resizer':
        method = Image.BILINEAR
        image_resizer_fn = functools.partial(
            preprocessor.resize_image,
            new_height=300,
            new_width=300,
            method=method)
    else:
        raise ValueError(
            'Invalid image resizer option: \'%s\'.' % image_resizer_config)

    def grayscale_image_resizer(image):
        [resized_image, resized_image_shape] = image_resizer_fn(image)
        grayscale_image = preprocessor.rgb_to_gray(resized_image)
        grayscale_image_shape = np.concatenate([resized_image_shape[:-1], [1]], 0)
        return [grayscale_image, grayscale_image_shape]

    return functools.partial(grayscale_image_resizer)
