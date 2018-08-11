"""Helper functions to access TensorShape values.

The rank 4 array_shape must be of the form [batch_size, height, width, depth].
"""

import numpy as np


def get_batch_size(array_shape):
    """Returns batch size from the tensor shape.

    Args:
      array_shape: A rank 4 TensorShape.

    Returns:
      An integer representing the batch size of the tensor.
    """
    assert (np.ndim(array_shape), 4)
    return array_shape[0]


def get_height(array_shape):
    """Returns height from the tensor shape.

    Args:
      array_shape: A rank 4 TensorShape.

    Returns:
      An integer representing the height of the tensor.
    """
    assert (np.ndim(array_shape), 4)
    return array_shape[1]


def get_width(array_shape):
    """Returns width from the tensor shape.

    Args:
      array_shape: A rank 4 TensorShape.

    Returns:
      An integer representing the width of the tensor.
    """
    assert (np.ndim(array_shape), 4)
    return array_shape[2]


def get_depth(array_shape):
    """Returns depth from the tensor shape.

    Args:
      array_shape: A rank 4 TensorShape.
  
    Returns:
      An integer representing the depth of the tensor.
    """
    assert (np.ndim(array_shape), 4)
    return array_shape[3].value
