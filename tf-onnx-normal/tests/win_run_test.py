import tensorflow as tf
import numpy as np
from tensorflow.core.framework import graph_pb2
import PIL.Image

def get_beach(name, path, shape):
    """Get beach image as input."""
    resize_to = shape[1:3]
    img = PIL.Image.open(path)
    img = img.resize(resize_to, PIL.Image.ANTIALIAS)
    img_np = np.array(img).astype(np.float32)
    img_np = img_np.reshape(shape)
    img_np = img_np / 127.5 - 1
    return {name: img_np}


def run_tensorflow(sess, inputs, out_tensor):
    print('run_tensorflow(): so we have a reference output')
    """Run model on tensorflow so we have a referecne output."""
    feed_dict = {}
    for k, v in inputs.items():
        k = sess.graph.get_tensor_by_name(k)
        feed_dict[k] = v

    result = sess.run(out_tensor, feed_dict=feed_dict)
    print("result:", result[0].shape)

    if result[0].shape == (1, 1, 1, 1001) or result[0].shape == (1, 1, 1, 1000):
        print(" result.shape:", result[0][0][0].shape)
        index = np.argmax(result[0][0][0], axis=1)
        print("\t result[0][0][0][index]:", result[0][0][0][0][index])
        print("\t index:", index)
    if result[0].shape == (1, 1000) or result[0].shape == (1, 1001):
        print(" result.shape:", result[0].shape)
        index = np.argmax(result[0], axis=1)
        print("\t result[0][0][index]:", result[0][0][index])
        print("\t index:", index)

    return result


def main():

    model_path = "E:\\Intenginetech\\tf-onnx\\tests\\models\\inception_v3\\inception_v3_2016_08_28_frozen.pb"
    graph_def = graph_pb2.GraphDef()
    print("model_path:", model_path)
    with open(model_path, "rb") as f:
        graph_def.ParseFromString(f.read())
    g = tf.import_graph_def(graph_def, name='')

    input_name = "input:0"
    path = "E:\\Intenginetech\\tf-onnx-normal\\tests\\dog.jpg"
    shape = [1, 299, 299, 3]
    out_tensor = ["InceptionV3/Predictions/Softmax:0"]
    inputs = get_beach(input_name,path, shape)
    with tf.Session(graph=g) as sess:
        tf_results = run_tensorflow(sess, inputs, out_tensor)
        print("done")


if __name__ == "__main__":
    main()
