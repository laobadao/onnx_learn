import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.tools.graph_transforms import TransformGraph
from tf2onnx import utils


def tf_optimize(sess, inputs, outputs, graph_def):
    # print("tf_optimize begin")
    """Optimize tensorflow graph for inference."""
    transforms = [
        "fold_constants(ignore_errors=true)",
        "fold_batch_norms",
        "fold_old_batch_norms",

    ]
    # TODO 这俩 在 研究 研究
    needed_names = [utils.node_name(i) for i in inputs] + [utils.node_name(i) for i in outputs]
    print("---------------needed_names:", needed_names)
    graph_def = graph_util.extract_sub_graph(graph_def, needed_names)

    print("extract_sub_graph done")

    graph_def = TransformGraph(graph_def, inputs, outputs, transforms)

    print("TransformGraph done")
    return graph_def


def main():
    inputs = ['image_tensor:0']
    outputs = ['Postprocessor/ExpandDims_1:0', 'Postprocessor/raw_box_scores:0']

    frozen_file = "E:\\models\\ssd_mobilenet_v1_qu\\frozen_inference_graph.pb"

    graph_def = tf.GraphDef()

    with tf.gfile.FastGFile(frozen_file, 'rb') as f:
        graph_def.ParseFromString(f.read())
    print("args.inputs:", inputs)
    print("args.outputs:", outputs)

    graph_def = tf_optimize(None, inputs, outputs, graph_def)

    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(graph_def, name='')

    ops1 = tf_graph.get_operations()

    tensor_names = []

    outputs_shape = {}

    for node in ops1:
        for out in node.outputs:
            tensor_names.append(node.name)

    print("len(tensor_names):", len(tensor_names))


    print("len(ops1):", len(ops1))

    image_tensor = tf_graph.get_tensor_by_name('image_tensor:0')

    print("image_tensor before:", image_tensor.shape.as_list())

    need_set_shape = False

    tf.slice()

    INVALID_SHAPE = {None, -1}

    for item in image_tensor.shape.as_list():
        print("type(item):", type(item))
        if item in INVALID_SHAPE:
            print("need set shape item:", item)
            need_set_shape = True
            break
    # if need_set_shape:

        # with tf.Session(graph=tf_graph) as sess1:
        #     ops = sess1.graph.get_operations()
        #
        #     sess1.run(tf.global_variables_initializer())
        #
        #     middle_output = sess1.run(preprocessor_tensor, feed_dict={image_tensor: image_value})




    # image_tensor.set_shape([1, 300, 300, 3])
    # print("image_tensor after:", image_tensor)
    # image_tensor1 = tf_graph.get_tensor_by_name('image_tensor:0')
    # print("image_tensor1 get:", image_tensor1)

    with tf.Session(graph=tf_graph) as sess:
        ops = sess.graph.get_operations()
        print("len(ops):", len(ops))

        # for op in ops:
        #     print(op.name)


if __name__ == "__main__":
    main()
