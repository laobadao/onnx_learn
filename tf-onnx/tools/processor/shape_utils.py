"""Utils used to manipulate ndarray shapes."""

import numpy as np

from tools.processor import static_shape


def _is_ndarray(t):
    """Returns a boolean indicating whether the input is a ndarray.

    Args:
      t: the input to be tested.

    Returns:
      a boolean that indicates whether t is a ndarray.
    """
    return isinstance(t, (np.ndarray,))


def _set_dim_0(t, d0):
    """Sets the 0-th dimension of the input ndarray.
  
    Args:
      t: the input array, assuming the rank is at least 1.
      d0: an integer indicating the 0-th dimension of the input ndarray.
  
    Returns:
      the array t with the 0-th dimension set.
    """
    t_shape = list(t.shape)
    t_shape[0] = d0
    t.reshape(t_shape)
    return t


def pad_array(t, length):
    """Pads the input array with 0s along the first dimension up to the length.

    Args:
      t: the input array, assuming the rank is at least 1.
      length: a array of shape [1]  or an integer, indicating the first dimension
        of the input array t after padding, assuming length <= t.shape[0].

    Returns:
      padded_t: the padded array, whose first dimension is length. If the length
        is an integer, the first dimension of padded_t is set to length
        statically.
    """
    t_rank = np.ndim(t)
    t_shape = t.shape
    t_d0 = t_shape[0]
    pad_d0 = t.expand_dims(length - t_d0, 0)
    if t_rank > 1:
        pad_shape = np.concatenate((pad_d0, t_shape[1:]), 0)
    else:
        pad_shape = np.expand_dims(length - t_d0, 0)
    padded_t = np.concatenate((t, np.zeros(pad_shape, dtype=t.dtype)), 0)
    if not _is_ndarray(length):
        padded_t = _set_dim_0(padded_t, length)
    return padded_t


def gather_numpy(t, axis, index):
    """
    Gathers values along an axis specified by axis.
    For a 3-D tensor the output is specified by:
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    :param axis: The axis along which to index
    :param index: A tensor of indices of elements to gather
    :return: tensor of gathered values
    """
    idx_xsection_shape = index.shape[:axis] + index.shape[axis + 1:]
    self_xsection_shape = t.shape[:axis] + t.shape[axis + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError("Except for dimension " + str(axis) +
                         ", all dimensions of index and self should be the same size")
    if index.dtype != np.dtype('int_'):
        raise TypeError("The values of index must be integers")
    data_swaped = np.swapaxes(t, 0, axis)
    index_swaped = np.swapaxes(index, 0, axis)
    gathered = np.choose(index_swaped, data_swaped)
    return np.swapaxes(gathered, 0, axis)


def clip_array(t, length):
    """Clips the input array along the first dimension up to the length.

    Args:
      t: the input array, assuming the rank is at least 1.
      length: a array of shape [1]  or an integer, indicating the first dimension
        of the input array t after clipping, assuming length <= t.shape[0].

    Returns:
      clipped_t: the clipped array, whose first dimension is length. If the
        length is an integer, the first dimension of clipped_t is set to length
        statically.
    """
    clipped_t = gather_numpy(t, axis=0, index=np.arange(length))
    if not _is_ndarray(length):
        clipped_t = _set_dim_0(clipped_t, length)
    return clipped_t


def pad_or_clip_array(t, length):
    """Pad or clip the input ndarray along the first dimension.

    Args:
      t: the input array, assuming the rank is at least 1.
      length: a array of shape [1]  or an integer, indicating the first dimension
        of the input array t after processing.

    Returns:
      processed_t: the processed array, whose first dimension is length. If the
        length is an integer, the first dimension of the processed array is set
        to length statically.
    """
    if t.shape[0] > length:
        processed_t = clip_array(t, length)
    else:
        processed_t = pad_array(t, length)
    if not _is_ndarray(length):
        processed_t = _set_dim_0(processed_t, length)
    return processed_t


def combined_static_and_dynamic_shape(t):
    """Returns a list containing static and dynamic values for the dimensions.

    Returns a list of static and dynamic values for shape dimensions. This is
    useful to preserve static shapes when available in reshape operation.

    Args:
      t: A ndarray of any type.

    Returns:
      A list of size ndarray.shape.ndims containing integers or a scalar ndarray.
    """
    static_ndarray_shape = list(t.shape)
    dynamic_ndarray_shape = t.shape
    combined_shape = []
    for index, dim in enumerate(static_ndarray_shape):
        if dim is not None:
            combined_shape.append(dim)
        else:
            combined_shape.append(dynamic_ndarray_shape[index])
    return combined_shape


def static_or_dynamic_map_fn(fn, elems, dtype=None):
    """Runs map_fn as a (static) for loop when possible.

    This function rewrites the map_fn as an explicit unstack input -> for loop
    over function calls -> stack result combination.  This allows our graphs to
    be acyclic when the batch size is static.
    For comparison, see https://www.ndarrayflow.org/api_docs/python/tf/map_fn.

    Note that `static_or_dynamic_map_fn` currently is not *fully* interchangeable
    with the default tf.map_fn function as it does not accept nested inputs (only
    ndarrays or lists of ndarrays).  Likewise, the output of `fn` can only be a
    ndarray or list of ndarrays.

    TODO(jonathanhuang): make this function fully interchangeable with tf.map_fn.

    Args:
      fn: The callable to be performed. It accepts one argument, which will have
        the same structure as elems. Its output must have the
        same structure as elems.
      elems: A ndarray or list of ndarrays, each of which will
        be unpacked along their first dimension. The sequence of the
        resulting slices will be applied to fn.
      dtype:  (optional) The output type(s) of fn. If fn returns a structure of
        ndarrays differing from the structure of elems, then dtype is not optional
        and must have the same structure as the output of fn.
      parallel_iterations: (optional) number of batch items to process in
        parallel.  This flag is only used if the native tf.map_fn is used
        and defaults to 32 instead of 10 (unlike the standard tf.map_fn default).
      back_prop: (optional) True enables support for back propagation.
        This flag is only used if the native tf.map_fn is used.

    Returns:
      A ndarray or sequence of ndarrays. Each ndarray packs the
      results of applying fn to ndarrays unpacked from elems along the first
      dimension, from first to last.
    Raises:
      ValueError: if `elems` a ndarray or a list of ndarrays.
      ValueError: if `fn` does not return a ndarray or list of ndarrays
    """
    if isinstance(elems, list):
        for elem in elems:
            if not isinstance(elem, np.ndarray):
                raise ValueError('`elems` must be a ndarray or list of ndarrays.')

        elem_shapes = [list(elem.shape) for elem in elems]
        # Fall back on tf.map_fn if shapes of each entry of `elems` are None or fail
        # to all be the same size along the batch dimension.
        for elem_shape in elem_shapes:
            if (not elem_shape or not elem_shape[0]
                    or elem_shape[0] != elem_shapes[0][0]):

                return tf.map_fn(fn, elems, dtype)

        arg_tuples = zip(*[np.split(elem, elem.shape[0]) for elem in elems])
        outputs = [fn(arg_tuple) for arg_tuple in arg_tuples]
    else:
        if not isinstance(elems, np.ndarray):
            raise ValueError('`elems` must be a ndarray or list of ndarrays.')
        elems_shape = list(elems.shape)
        if not elems_shape or not elems_shape[0]:
            return tf.map_fn(fn, elems, dtype)



        outputs = [fn(arg) for arg in np.split(elems, elems.shape[0])]

    # Stack `outputs`, which is a list of ndarrays or list of lists of ndarrays
    if all([isinstance(output, np.ndarray) for output in outputs]):
        return np.stack(outputs)
    else:
        if all([isinstance(output, list) for output in outputs]):
            if all([all(
                    [isinstance(entry, np.ndarray) for entry in output_list])
                for output_list in outputs]):
                return [np.stack(output_tuple) for output_tuple in zip(*outputs)]
    raise ValueError('`fn` should return a ndarray or a list of ndarrays.')


def check_min_image_dim(min_dim, image_ndarray):
    """Checks that the image width/height are greater than some number.

    This function is used to check that the width and height of an image are above
    a certain value. If the image shape is static, this function will perform the
    check at graph construction time. Otherwise, if the image shape varies, an
    Assertion control dependency will be added to the graph.

    Args:
      min_dim: The minimum number of pixels along the width and height of the
               image.
      image_ndarray: The image ndarray to check size for.

    Returns:
      If `image_ndarray` has dynamic size, return `image_ndarray` with a Assert
      control dependency. Otherwise returns image_ndarray.

    Raises:
      ValueError: if `image_ndarray`'s' width or height is smaller than `min_dim`.
    """
    image_shape = image_ndarray.shape
    image_height = static_shape.get_height(image_shape)
    image_width = static_shape.get_width(image_shape)

    # TODO (ZJ)：ssd image h , w != None ，but Faster rcnn maybe 。modify later

    if image_height is None or image_width is None:
        shape_assert = tf.Assert(
            tf.logical_and(tf.greater_equal(tf.shape(image_tensor)[1], min_dim),
                           tf.greater_equal(tf.shape(image_tensor)[2], min_dim)),
            ['image size must be >= {} in both height and width.'.format(min_dim)])
        with tf.control_dependencies([shape_assert]):
            return tf.identity(image_tensor)

    if image_height is None or image_width is None:
        x = image_ndarray.shape[1] >= min_dim
        y = image_ndarray.shape[2] >= min_dim
        assert (x & y, ['image size must be >= {} in both height and width.'.format(min_dim)])

        return np.identity(image_ndarray)

    if image_height < min_dim or image_width < min_dim:
        raise ValueError(
            'image size must be >= %d in both height and width; image dim = %d,%d' %
            (min_dim, image_height, image_width))

    return image_ndarray


def assert_shape_equal(shape_a, shape_b):
    """Asserts that shape_a and shape_b are equal.

    If the shapes are static, raises a ValueError when the shapes
    mismatch.

    If the shapes are dynamic, raises a tf InvalidArgumentError when the shapes
    mismatch.

    Args:
      shape_a: a list containing shape of the first ndarray.
      shape_b: a list containing shape of the second ndarray.

    Returns:
      Either a tf.no_op() when shapes are all static and a tf.assert_equal() op
      when the shapes are dynamic.

    Raises:
      ValueError: When shapes are both static and unequal.
    """
    if (all(isinstance(dim, int) for dim in shape_a) and
            all(isinstance(dim, int) for dim in shape_b)):
        if shape_a != shape_b:
            raise ValueError('Unequal shapes {}, {}'.format(shape_a, shape_b))
        else:
            return 0
    else:
        return np.testing.assert_equal(shape_a, shape_b)


def assert_shape_equal_along_first_dimension(shape_a, shape_b):
    """Asserts that shape_a and shape_b are the same along the 0th-dimension.

    If the shapes are static, raises a ValueError when the shapes
    mismatch.

    If the shapes are dynamic, raises a tf InvalidArgumentError when the shapes
    mismatch.

    Args:
      shape_a: a list containing shape of the first ndarray.
      shape_b: a list containing shape of the second ndarray.

    Returns:
      Either a tf.no_op() when shapes are all static and a tf.assert_equal() op
      when the shapes are dynamic.

    Raises:
      ValueError: When shapes are both static and unequal.
    """
    if isinstance(shape_a[0], int) and isinstance(shape_b[0], int):
        if shape_a[0] != shape_b[0]:
            raise ValueError('Unequal first dimension {}, {}'.format(
                shape_a[0], shape_b[0]))
        else:
            # tf.no_op
            return 0
    else:
        return np.testing.assert_equal(shape_a[0], shape_b[0])
