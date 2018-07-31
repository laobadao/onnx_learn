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

    with tf.Session(graph=tf_graph) as sess:
        ops = sess.graph.get_operations()
        print("len(ops):", len(ops))

        for op in ops:
            print(op.name)


if __name__ == "__main__":
    main()
