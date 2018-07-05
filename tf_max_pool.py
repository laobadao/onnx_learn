import tensorflow as tf

X = tf.constant([
    [[1.0, 2.0, 3.0, 4.0],
     [5.0, 6.0, 7.0, 8.0],
     [9.0, 10.0, 11.0, 12.0],
     [13.0, 14.0, 15.0, 16.0]]

])
X = tf.reshape(X, [1, 4, 4, 1])

pooling1 = tf.nn.max_pool(X, [1, 1, 1, 1], [1, 2, 2, 1], padding='SAME')
with tf.Session() as sess:
    print("image:")
    image1 = sess.run(X)
    print(image1)
    print("reslut:")
    result1 = sess.run(pooling1)
    print(result1)
