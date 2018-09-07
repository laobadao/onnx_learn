import tensorflow as tf
import numpy as np
from PIL import Image
import os

np.set_printoptions(threshold=np.inf)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def build_ssd_preprocessor(img_path):
    image = Image.open(img_path)
    #image = image.resize((500, 500), Image.BILINEAR)
    #image.save("detection_4.bmp", 'BMP', quality=95)
    # import cv2
    # pic = cv2.imread(img_path)
    # print("pic 1:", type(pic))
    # pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
    # print("pic 2:", type(pic))
    # image_np = cv2.resize(pic, (300, 300), interpolation=cv2.INTER_LINEAR)

    image = image.resize((300, 300), Image.BILINEAR)
    image_np = load_image_into_numpy_array(image)
    print("image_np:", image_np.shape)
    resized_inputs = image_np.astype(np.float32)
    image_np_expanded = np.expand_dims(resized_inputs, axis=0)
    print("image_np_expanded:", image_np_expanded.shape)

    final_result = (2.0 / 255.0) * image_np_expanded - 1.0
    # print("final_result:", final_result.shape)

    return final_result


# x_shape = [1, 15, 20, 2]
# x_new_size = [1, 300, 300, 3]
# stop = 1 + np.prod(x_shape)
# print("stop:", stop)
# x_val = np.arange(1, stop).astype("float32").reshape(x_new_size)
# print("x_val:", x_val.shape)

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection_500.bmp")
np_result = build_ssd_preprocessor(path)


image = Image.open(path)
image_np = load_image_into_numpy_array(image)
image_np_expanded = np.expand_dims(image_np, axis=0)

with tf.Session() as sess:

    placeholder = tf.placeholder(dtype=tf.uint8, shape=[1, None, None, 3])
    image_np_tf = tf.to_float(placeholder)
    resize_image = tf.image.resize_images(image_np_tf, tf.stack([300, 300]))

    tf.layers.dense()

    result = sess.run(resize_image, feed_dict={placeholder: image_np_expanded})
    print("result.shape:", result.shape)

    resized_inputs = (2.0 / 255.0) * result - 1.0
    # f = open("tf_result.txt", "w")
    #
    # print("resized_inputs:", resized_inputs, file=f)
    print("sub:", np_result - resized_inputs)













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

#
# elems = np.array([1, 2, 3, 4, 5, 6])
# squares = tf.map_fn(lambda x: x * x, elems)
#
# elems1 = (np.array([1, 2, 3]), np.array([-1, 1, -1]))
# alternate = tf.map_fn(lambda x: x[0] * x[1], elems1, dtype=tf.int64)
#
# elems2 = np.array([1, 2, 3])
# alternates = tf.map_fn(lambda x: (x, -x), elems2, dtype=(tf.int64, tf.int64))
#
# with tf.Session() as sess:
#     print(sess.run(squares))
#     print(sess.run(alternate))
#     print(sess.run(alternates))
# squares == [1, 4, 9, 16, 25, 36]
# alternate == [-1, 2, -3]

# alternates[0] == [1, 2, 3]
# alternates[1] == [-1, -2, -3]

# elems = np.array([1, 2, 3, 4, 5, 6])
#
# print(elems.dtype)
# elems = elems.astype(np.float32)
# print("after:", elems.dtype)
#
# if elems.dtype != np.float32:
#
#     raise ValueError('`preprocess` expects a float32 ndarray')


