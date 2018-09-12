from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import onnx

import PIL.Image
import numpy as np
import tensorflow as tf
import tf2onnx
import yaml
from tf2onnx.tfonnx import process_tf_graph
from PIL import Image, ImageDraw
import os
import caffe2.python.onnx.backend
import cv2
import h5py

def run_caffe2(model_proto, inputs):
    """Run test again caffe2 backend."""
    print("run_caffe2()---------------------------------------------")
    prepared_backend, ws = caffe2.python.onnx.backend.prepare(model_proto)
    #open("init.pb", 'w').write(str(prepared_backend.init_net))

    result = prepared_backend.run(inputs)
    if ws != None:
        f = h5py.File("output.hdf5", "w")
        for node in model_proto.graph.node:
            #print(node.output)
            data = ws.FetchBlob(node.output[0])[0]
            print(node.output, data.shape )
            #print(data[0][4][4])
            f[node.output[0]] = data[0]
            break
            #data = data.transpose( ( 2, 0, 1 ) )
            #print(node.output, np.sum(data))
            #f[node.output[0]] = data 


    if result[0].shape == (1, 1000):
        print(" result:", result)
        #print(" result[0].shape:", result[0].shape)
        index = np.argmax(result[0], axis=1)
        print(" index:", index)
        print("result[0[0][index]:", result[0][0][index])
    # resnet v1 50 的版本需要 这样去检测 index 它的 shape 维数比较多 1000 类
    return result

def get_beach():
    """Get beach image as input."""
    # for name, shape in inputs.items():
    #     print("name:", name, "shape:", shape)
    #     break
    name = "gpu_0/data_0"
    shape = [1, 3, 224, 224]
    resize_to = shape[2:4]
    print("resize_to:", resize_to)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "elephant.jpg")
    print("path:", path)
    #img = PIL.Image.open(path)
    img = cv2.imread(path)
    img = cv2.resize(img,(224,224))
    img = img/127.5 - 1
    #img = img[ :, :, np.argsort([2, 1, 0]) ]
    img = img.transpose( ( 2, 0, 1 ) )
    #img = np.zeros([224, 224, 3])
    #img[10, 10, :] = 1
    #img = img.transpose( (2, 0, 1) )

    #print(img)
    #f = h5py.File("img.hdf5", "w")
    #f["data"] = img  

    img = img[np.newaxis, :, :, :].astype(np.float32)
    #img_np = img_np / 255
    #print("img_np:", img_np)
    return {name: img}



if __name__ == "__main__":
    # Input
    # gpu_0 / data_0: float[1, 3, 224, 224]
    # Output
    # gpu_0 / softmax_1: float[1, 1000]

    model_proto = onnx.load('model.onnx') 
    #open("initializer.pb", 'wb').write( model_proto.graph.initializer)

    # create the input data
    inputs = get_beach()
    onnx_results = run_caffe2(model_proto, inputs)
