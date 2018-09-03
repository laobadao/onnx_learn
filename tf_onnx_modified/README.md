# tf2onnx & onnx2Keras & Keras2ISD

- Tf2onnx converts a TensorFlow graph to an ONNX graph.
- onnx2Keras converts onnx graph to a keras.json.
- Keras2ISD converts a keras.json to a ISD.json.

注:
1. 此版 onnx 中部分 op 为自定义格式,非官方 release 版。
2. 此版 Keras 中含有自定义 layer 或 activation, 非官方 release 版。

|版本| 修改日期 | 修改人员|备注|
|-------|-------|-------|
|ver 0.1| 2018-8-21| 赵君君|文档初稿|
|ver 0.1| 2018-8-22| 赵君君|添加部分对应 OP/layer 转换表格|
|ver 0.1| 2018-8-24| 赵君君|删除部分不支持的 op |
|ver 0.1| 2018-8-27| 赵君君|文档整理|
|ver 0.1| 2018-8-28| 赵君君|添加 部分 OP 合并为 Layer 的说明 |

# Support OS

- Linux
- Windows 10

# Support Tensorflow layer/op

说明:

- **TF op/layer**: Tensorflow frozen pb 中 OP 的 name.
- **onnx(modified) op**: 自定义版的 onnx 对应 OP 的 name.
- **Keras Layer**: 自定义版本的 Keras  中 layer 的 name.
- **Keras(tf backend)**: 以 tensorflow 为 backend run 后的结果。
- **ISD**: ISD 超级层 layer 的 name.

注:

- **@MN**: onnx 自定义 name 和 attrs
- **@MA**: onnx 自定义 attrs
- **@CL**: keras 自定义 Layer
- **@CA**: keras 自定义 Activation  


## Op / Layer 1 对 1 转换

| TF op/layer | onnx(modified) op | Keras Layer | Keras(tf backend)  |  ISD |
| :-----: | :-----:|:-----:|:-----: |
|  Conv2D| Conv | Conv2D| √ | |
|  DepthwiseConv2d| DepthwiseConv2d(@MN) | DepthwiseConv2D| √ | |
|  DepthwiseConv2dNative|DepthwiseConv2dNative(@MA)|DepthwiseConv2D |√ | |
| BatchNormalization | BatchNormalization | BatchNormalization| √| |
| FusedBatchNorm | BatchNormalization | BatchNormalization| √| |
| ReLU  | ReLU | relu| √|  |
| Relu6  |Relu6(@MN) |relu6(@CA) | √|  |
| Softmax | Softmax |softmax | √|  |
| Sigmoid |Sigmoid  |sigmoid |√ |  |
| Pad、PadV2  | Pad [1] | ZeroPadding2D| √|  |
| MaxPool、MaxPoolV2 |  MaxPool| MaxPooling2D| √|  |
| AvgPool |  AveragePool| AveragePooling2D| √|  |
| Flatten | Flatten | Flatten| √|  |
| Reshape | Reshape | Reshape|√ |  |
|BiasAdd、BiasAddV1  | Add_(@MN) | 注[3] 合并到 Conv2D 中| √|  |
| Add | Add | Add| √|  |
| AddN | Sum | Sum| √|  |
| Mean | ReduceMean | GlobalAveragePooling2D|√ |  |
| Concat、ConcatV2 | Concat | Concatenate| √|  |
| SpaceToDepth | SpaceToDepth | @CL| √|  |
|Squeeze  | Squeeze | @CL|√ |  |
| Mul | Mul |  | |  |
| Transpose |Transpose | Permute| |  |
| Conv2DBackpropInput |ConvTranspose |Conv2DTranspose | |  |
| StridedSlice、Slice |Slice | | |  |
| Abs |Abs | | |  |
|  |  GlobalMaxPool| GlobalMaxPooling2D| |  |
|  | PReLU | PReLU@CA| |  |
|  | ELU | ELU| |  |
|  | Tanh |Tanh  | |  |


## Op/Layer 多 对 1 转换

注: 以下表格内为多 op 转换为一个 layer 的转换对应关系。仅在以下模式中可以转换，若单独出现下面的 op (且非上面表格中声名支持的 op),则不予以支持。

| TF op/layer | onnx(modified) op | Keras Layer | Keras(tf backend) run | ISD |
|------|------|------|
|Mul,Maximum | LeakyRelu | LeakyReLU|√ |  |
|StridedSlice,ResizeBilinear,Pad| UpSampling2D(@MN)  | UpSampling2D| | |
| Add, Mul,Rsqrt,Add,Sub,Mul | BatchNormalization[6] | BatchNormalization | √| |
| Add, Mul,Mul,Rsqrt,Add,Sub,Mul | BatchNormalization[7] | BatchNormalization | √| |
| MatMul,BiasAdd| Dense | Dense| √| |
| Reshape,Pack,StridedSlice,Shape| Flatten | Flatten| √| | |


## 表示 input & weights 等 op

注：以下 op 一般用作输入，以及存储 weights,bias,scalar 等具体数值 的 op。

| TF op/layer | onnx(modified) op | Keras Layer | Keras(tf backend) run | ISD |
|------|------|------|
|Placeholder、PlaceholderV2、PlaceholderWithDefault | 注[4] | | |  |
|Identity  | 注[2] | | |  |
| Const、ConstV2 | 注[5] | | |  |

注：

[1]. Tensorflow 中 Pad op mode：取值为 "CONSTANT"、"REFLECT" 或 "SYMMETRIC"（不区分大小写）,我们仅支持 CONSTANT mode 且 value 为 0 的情况。

[2]. Identity 在转换时内部处理,不进行 OP 的转换,仅保存需要的 weights, bias, const, attr 等,但是对外(TF)支持。

[3]. Tensorflow 中的 BiasAdd 在 onnx 中命名为 Add_, 在向 Keras 转换时，合并到 Conv2D 中。

[4]. Tensorflow 中的 Placeholder、PlaceholderV2、PlaceholderWithDefault 一般作为 inputs 输入。

[5]. Tensorflow 中的  Const、ConstV2 一般存储 weights,bias,scalar, 等 const 类数据。

[6]. Tensorflow 相对比较旧的版本中实现 BatchNormalization 时，无 gamma 的实现方法。

[7]. Tensorflow 相对比较旧的版本中实现 BatchNormalization 时，有 gamma 的实现方法。

---

# Prerequisites

## Install TensorFlow
If you don't have tensorflow installed already, install the desired tensorflow build, for example:
```
pip3 install tensorflow
or
pip3 install tensorflow-gpu
```


## Python Version
We tested with tensorflow 1.5,1.6,1.7,1.8,1.9,1.10 and anaconda **3.5,3.6**.

# Installation

Once dependencies are installed, from the tensorflow-onnx folder call:

```
python3 setup.py install
```
tensorflow-onnx requires onnx-1.2.2 or better and will install/upgrade onnx if needed.


# 使用方法

为了可以转换一个 TensorFlow model, tf2onnx 需要一个  ```frozen TensorFlow graph``` and the 用户需要明确 graph 的 inputs and outputs 通过传递 input and output
names with ```--inputs INPUTS``` and ```--outputs OUTPUTS```.

使用的命令行:
```
python -m tf2onnx.convert --input SOURCE_FROZEN_GRAPH_PB\
    --inputs SOURCE_GRAPH_INPUTS\
    --outputs SOURCE_GRAPH_OUTPUS\
    [--middle_inputs SUB_GRAPH_INPUTS\
    [--middle_outputs SUB_GRAPH_OUTPUS\
    [--output TARGET_ONNX_GRAPH]\  
```

## Parameters 说明:
- input: frozen TensorFlow graph, which can be got with [freeze graph tool](#freeze_graph).
- output:  onnx 文件输出存储路径.
- inputs/outputs: Tensorflow graph 的 输入/输出 names, 可以通过 [summarize graph tool](#summarize_graph) 获得.
- middle_inputs:  基本出现在 tensorflow  object detection 的 frozen.pb 若模型中包含 resize 以及 归一化预处理 等操作，导致包含 控制流系列的 op ，这部分需要删减掉，如 ssd-mobile 和 faster-rcnn 等 需要充新定义中间的 输入节点名字。
- middle_outputs: 基本出现在 tensorflow  object detection 的 frozen.pb 若模型中包含后处理 Decode  NMS ROIPooling CropAndResize 等操作，导致包含控制流系列或其他我们暂不支持的 op ，这部分需要删减掉，如 ssd-mobile 和 faster-rcnn 等 需要充新定义中间的输出节点名字。


## Usage example :

### 简单模型的准换（仅含有 inputs/outputs 的转换）：
```
python -m tf2onnx.convert\
    --input tests/models/fc-layers/frozen.pb\
    --inputs X:0\
    --outputs output:0\
    --output tests/models/fc-layers/model.onnx\    
```
### 复杂模型的准换（含有 middle_inputs/middle_outputs 的转换）：

注： onnx 不支持 placeholder 中 shape 含有 -1 或 None 的情况。 所以若存在这种 shape，则可以使用 --inputs image_tensor:0[1,300,300,3] 转换的时候后面加上  [1,300,300,3] 来重写 shape。

```

python -m tf2onnx.convert\
    --input frozen_inference_graph.pb\
    --inputs image_tensor:0[1,300,300,3]\
    --outputs  detection_boxes:0,detection_scores:0,num_detections:0,detection_classes:0\
    --middle_inputs Preprocessor/sub:0\
    --middle_outputs BoxPredictor_0/BoxEncodingPredictor/BiasAdd:0,BoxPredictor_0/ClassPredictor/BiasAdd:0,BoxPredictor_1/BoxEncodingPredictor/BiasAdd:0,BoxPredictor_1/ClassPredictor/BiasAdd:0,BoxPredictor_2/BoxEncodingPredictor/BiasAdd:0,BoxPredictor_2/ClassPredictor/BiasAdd:0,BoxPredictor_3/BoxEncodingPredictor/BiasAdd:0,BoxPredictor_4/BoxEncodingPredictor/BiasAdd:0,BoxPredictor_4/ClassPredictor/BiasAdd:0,BoxPredictor_5/BoxEncodingPredictor/BiasAdd:0,BoxPredictor_5/ClassPredictor/BiasAdd:0 --output ssd_mobile_v1_q.onnx

```

## 使用 yaml 格式配置参数 使用 convert_models.py 进行转换
```
python tests/convert_models.py
usage: convert_models.py  [--config yaml-config] [--onnx-file onnx file path]

arguments:
  --config           yaml config file
  --onnx-file        onnx 格式文件的存储路径
```
```convert_models.py``` will convert the TensorFlow model.

You call it for example with:

```
python tests/convert_models.py --config tests/ssd_mobile.yaml  --onnx-file ssd_mobile
```

### yaml example:

```
ssd_mobile:
  model: tests/models/ssd_mobile/frozen_inference_graph.pb
  inputs:
    "image_tensor:0":[1, 300, 300, 3]
  outputs:
    - detection_boxes:0
    - detection_scores:0
    - num_detections:0
    - detection_classes:0
  middle_inputs:
    "Preprocessor/sub:0": [1, 300, 300, 3]
  force_input_shape: True
  middle_outputs:
    - BoxPredictor_0/BoxEncodingPredictor/BiasAdd:0
    - BoxPredictor_0/ClassPredictor/BiasAdd:0
    - BoxPredictor_1/BoxEncodingPredictor/BiasAdd:0
    - BoxPredictor_1/ClassPredictor/BiasAdd:0
    - BoxPredictor_2/BoxEncodingPredictor/BiasAdd:0
    - BoxPredictor_2/ClassPredictor/BiasAdd:0
    - BoxPredictor_3/BoxEncodingPredictor/BiasAdd:0
    - BoxPredictor_3/ClassPredictor/BiasAdd:0
    - BoxPredictor_4/BoxEncodingPredictor/BiasAdd:0
    - BoxPredictor_4/ClassPredictor/BiasAdd:0
    - BoxPredictor_5/BoxEncodingPredictor/BiasAdd:0
    - BoxPredictor_5/ClassPredictor/BiasAdd:0

```

### yaml 格式参数配置说明：

- model: pb 地址
- inputs: pb 中 inputs names
- outputs: pb 中 outputs names
- middle_inputs: [optional] 因部分 tensorflow 的 op 暂不支持转换，所以 pb 中若包含预处理及后处理部分的 模块内的 op 都需要截断，重新定义输入，输出节点。
- force_input_shape：[optional] True 则代表会重写转换后真正的输入的 shape ,会以 middle_inputs 中的 shape 为准。如 [1, 300, 300, 3]，一般出现在原 pb 模型中 op 的 tensor 保存的 shape 含有 -1 或 None 的情况。
- middle_outputs: [optional] 因部分 tensorflow 的 op 暂不支持转换，所以 pb 中若包含预处理及后处理部分的 模块内的 op 都需要截断，重新定义输入，输出节点。


## <a name="summarize_graph"></a>Tool to Get Graph Inputs & Outputs

用来找到 tensorflow model 中 graph 对应的输入输出的工具，开发者可以访问  TensorFlow's [summarize_graph](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms) tool, for example:
```
summarize_graph --in_graph=tests/models/fc-layers/frozen.pb
```

## Tool to find right middle_inputs & middle_outputs

如果你想查看模型中是否存在我们不支持的 op ,并想重新定义中间的输入输出节点 middle_inputs & middle_outputs。 我们目前建议可以使用这种方式查看。

 将 pb 中的 graph  使用 tf.summary.FileWriter 方法生成可视化的图，可在图中查看并确定中间的输入输出节点 middle_inputs & middle_outputs。

### 示例如下：
```
import tensorflow as tf
from tensorflow.python.platform import gfile
model = 'frozen_model_vgg_16.pb'
graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
graph_def.ParseFromString(gfile.FastGFile(model, 'rb').read())
tf.import_graph_def(graph_def, name='')
summaryWriter = tf.summary.FileWriter('log/', graph)
```
### 使用 tensorboard 查看：
```
tensorboard --logdir log
```

## <a name="freeze_graph"></a>Tool to Freeze Graph

1. The TensorFlow tool to freeze the graph is [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py).

For example:
```
python -m tensorflow.python.tools.freeze_graph \
    --input_graph=my_checkpoint_dir/graphdef.pb \
    --input_binary=true \
    --input_names=input:0 \
    --output_node_names=output:0 \
    --input_checkpoint=my_checkpoint_dir \
    --output_graph=tests/models/fc-layers/frozen.pb
```

2. 或者在网络模型代码中进行 freeze

参考示例：

```
from tensorflow.python.framework import graph_util

output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
     sess,
     sess.graph_def,
     ['detection_boxes','detection_scores','num_detections','detection_classes']  # 如果有多个输出节点，以逗号隔开
       )

PB_DIR = "frozen_pb_dir"
PB_NAME = "new_frozen_pb.pb"

output_graph = os.path.join(PB_DIR, PB_NAME)  # PB模型保存路径

with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
    f.write(output_graph_def.SerializeToString())  # 序列化输出

print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点
```

3. 用 ckpt meta 等保存的 图以及权重的文件，合成并 freeze 成 pb

参考代码：

```
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import tensorflow as tf
import os


def freeze_session(sess, keep_var_names=None, output_names=None, clear_devices=True):
    """Freezes the state of a session into a pruned computation graph."""
    output_names = [i.replace(":0", "") for i in output_names]
    graph = sess.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(sess, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


def main():
    dir_name = "frozen"
    saver = tf.train.import_meta_graph('./checkpointdir/model.ckpt.meta', clear_devices=True)
    with tf.Session() as sess:
        saver.restore(sess, './checkpointdir/model.ckpt')
        frozen_graph = freeze_session(sess, output_names=["Sigmoid:0"])
        tf.train.write_graph(frozen_graph, dir_name, "frozen.pb", as_text=False)
    model_path = os.path.join(dir_name, "frozen.pb")
    print("model_path:", model_path)


if __name__ == "__main__":
    main()


```
