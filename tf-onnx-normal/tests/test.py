import numpy as np
from onnx import numpy_helper
from onnx_pb2 import * 
import os
import cv2
import h5py


path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "elephant.jpg")
img = cv2.imread(path)
img = cv2.resize(img,(224,224))
img = img/127.5 - 1

data = np.zeros([230, 230, 3])
data[3:227, 3:227, ] = img

#data = data.transpose( ( 2, 0, 1 ) )

pb = ModelProto()
pb.ParseFromString(
    open("model.onnx", 'rb').read()
)

init_array = pb.graph.initializer
for w in init_array:
    if w.name == "gpu_0/res_conv1_bn_riv_0":
        weight = numpy_helper.to_array(w)
        print(weight)

#weight = numpy_helper.to_array(init_array[0])

#first_weight = first_weight.transpose( ( 1, 2, 0 ) )


#f = h5py.File("output.hdf5", "w")
#f["weight"] = first_weight[0:7, 0:7, :]
#f["data"] = data[0:7,0:7,:]

#num = np.sum( first_weight[0:7, 0:7, :] * data[0:7,0:7,:] )
#print(num)            




    




