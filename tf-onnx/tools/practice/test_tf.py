import tensorflow as tf
import numpy as np

# a1 = np.array([[1, 4], [2, 5], [3, 6]])
#
# b1 = np.split(a1, a1.shape[0], axis=0)
#
# print("b1:", b1)
#
# a = tf.constant([1, 2, 3])
# b = tf.constant([4, 5, 6])
# c = tf.stack([a, b], axis=1)
# d = tf.unstack(c, axis=0)
# e = tf.unstack(c, axis=1)
# print(c.get_shape())
# with tf.Session() as sess:
#     print(sess.run(c))
#     print(sess.run(d))
#     print(sess.run(e))
#
#


elems = np.array([1, 2, 3, 4, 5, 6])
squares = tf.map_fn(lambda x: x * x, elems)

elems1 = (np.array([1, 2, 3]), np.array([-1, 1, -1]))
alternate = tf.map_fn(lambda x: x[0] * x[1], elems1, dtype=tf.int64)

elems2 = np.array([1, 2, 3])
alternates = tf.map_fn(lambda x: (x, -x), elems2, dtype=(tf.int64, tf.int64))

with tf.Session() as sess:
    print(sess.run(squares))
    print(sess.run(alternate))
    print(sess.run(alternates))
# squares == [1, 4, 9, 16, 25, 36]
# alternate == [-1, 2, -3]

# alternates[0] == [1, 2, 3]
# alternates[1] == [-1, -2, -3]


