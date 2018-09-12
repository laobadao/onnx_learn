# Operator Schemas

|版本| 修改日期 | 修改人员|备注|
|-------|-------|-------|
|ver 0.1| 2018-8-29| 赵君君| Operator Schemas 文档初稿|
|ver 0.1| 2018-8-30| 赵君君| Operator Schemas 完善|

注:

1. 数据输入，输出格式全部为 NHWC, 不会出现 Transpose 对 NHWC 进行转换。
2. 基本上修改了大部分原 onnx 格式规范，所有 op 的 spec 以本文档为标准。

  * <a href="#Abs">Abs</a>
  * <a href="#Add">Add</a>
  * <a href="#AveragePool">AveragePool</a>
  * <a href="#BatchNormalization">BatchNormalization</a>
  * <a href="#Concat">Concat</a>
  * <a href="#Constant">Constant</a>
  * <a href="#Conv">Conv</a>
  * <a href="#ConvTranspose">ConvTranspose</a>
  * <a href="#Elu">Elu</a>
  * <a href="#Flatten">Flatten</a>
  * <a href="#GlobalAveragePool">GlobalAveragePool</a>
  * <a href="#GlobalMaxPool">GlobalMaxPool</a>
  * <a href="#MaxPool">MaxPool</a>
  * <a href="#Mul">Mul</a>
  * <a href="#PRelu">PRelu</a>
  * <a href="#Pad">Pad</a>
  * <a href="#ReduceMean">ReduceMean</a>
  * <a href="#Relu">Relu</a>
  * <a href="#Reshape">Reshape</a>
  * <a href="#Sigmoid">Sigmoid</a>  
  * <a href="#Softmax">Softmax</a>
  * <a href="#SpaceToDepth">SpaceToDepth</a>
  * <a href="#Squeeze">Squeeze</a>
  * <a href="#Sum">Sum</a>
  * <a href="#Tanh">Tanh</a>
  * <a href="#Transpose">Transpose</a>
  * <a href="#Slice">Slice</a>
  * <a href="#MatMul">MatMul</a>
  * <a href="#Max">Max</a>
  * <a href="#Scale">Scale</a>
  * <a href="#AddBias">AddBias</a>
  * <a href="#UpSampling2D">UpSampling2D</a>  
  * <a href="#Relu6">Relu6</a>
  * <a href="#LeakyRelu">LeakyRelu</a>
  * <a href="#DepthwiseConv2d">DepthwiseConv2d</a>
  * <a href="#Dense">Dense</a>


## model.onnx (modified)
### <a name="Abs"></a><a name="abs">Abs</a>

  Absolute takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the absolute is, y = abs(x), is applied to
  the tensor elementwise.

#### Inputs

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input tensor</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output tensor</dd>
</dl>


### <a name="Add"></a><a name="add">Add</a>

  Performs element-wise binary addition (we do not support Numpy-style broadcasting).

#### Inputs

<dl>
<dt><tt>A</tt> : T</dt>
<dd>First operand.</dd>
<dt><tt>B</tt> : T</dt>
<dd>Second operand.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>C</tt> : T</dt>
<dd>Result, has same element type as two inputs</dd>
</dl>


### <a name="AveragePool"></a><a name="averagepool">AveragePool</a>

  AveragePool consumes an input tensor X and applies average pooling across the
   the tensor according to kernel sizes, stride sizes, and pad lengths.
   average pooling consisting of computing the average on all values of a
   subset of the input tensor according to the kernel size and downsampling the
   data into the output tensor Y for further processing.

#### Attributes

<dl>
<dt><tt>kernel_shape</tt> : list of ints (required)</dt>
<dd>The size of the kernel along each axis.</dd>
<dt><tt>pads</tt> : string</dt>
<dd>"SAME" or "VALID"</dd>
<dt><tt>strides</tt> : list of ints</dt>
<dd>Stride along each axis. If not present, the stride defaults to 1 along each axis.</dd>
</dl>

#### Inputs

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input data tensor from the previous operator; dimensions for image case are (N x H x W x C ), where N is the batch size, H and W are the height and the width of the data, and C is the number of channels. For non image case, the dimensions are in the form of (N  x D1 x D2 ... Dn x C), where N is the batch size. Optionally, if dimension denotation is in effect, the operation expects the input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_FEATURE, DATA_FEATURE ..., DATA_CHANNEL].</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output data tensor from average or max pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad type. </dd>
</dl>


### <a name="BatchNormalization"></a><a name="batchnormalization">BatchNormalization</a>

  Carries out batch normalization as described in the paper
  https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
  there are multiple cases for the number of outputs, which we list below:

  Output case #1: Y, mean, var, saved_mean, saved_var (training mode, we do not support this mode)

  Output case #2: Y (test mode, only support this)


#### Attributes

<dl>
<dt><tt>epsilon</tt> : float</dt>
<dd>The epsilon value to use to avoid division by zero, default is 1e-5f.</dd>
<dt><tt>momentu</tt> : float</dt>
<dd>Factor used in computing the running mean and variance.e.g., running_mean = running_mean * momentum + mean * (1 - momentum), default is 0.9f.</dd>
<dt><tt>spatial</tt> : int</dt>
<dd>If true, compute the mean and variance across all spatial elements If false, compute the mean and variance across per feature.Default is 1.</dd>
</dl>

#### Inputs

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.</dd>
<dt><tt>scale** (optional)</tt> : T</dt>
<dd> The scale as a 1-dimensional tensor of size C to be applied to the output.  some old version tensorflow don't have scale.</dd>
<dt><tt>B</tt> : T</dt>
<dd>The bias as a 1-dimensional tensor of size C to be applied to the output.</dd>
<dt><tt>mean</tt> : T</dt>
<dd>The running mean (training) or the estimated mean (testing) as a 1-dimensional tensor of size C.</dd>
<dt><tt>var</tt> : T</dt>
<dd>The running variance (training) or the estimated variance (testing) as a 1-dimensional tensor of size C.</dd>
</dl>

#### Outputs (1 - 5)

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>The output tensor of the same shape as X.</dd>
<dt><tt>mean</tt> (optional) : T</dt>
<dd>The running mean after the BatchNormalization operator. we usually delete this.</dd>
<dt><tt>var</tt> (optional) : T</dt>
<dd>The running variance after the BatchNormalization operator.we usually delete this.</dd>
<dt><tt>saved_mean</tt> (optional) : T</dt>
<dd>Saved mean used during training to speed up gradient computation.we usually delete this.</dd>
<dt><tt>saved_var</tt> (optional) : T</dt>
<dd>Saved variance used during training to speed up gradient computation.we usually delete this.</dd>
</dl>


### <a name="Concat"></a><a name="concat">Concat</a>

  Concatenate a list of tensors into a single tensor

#### Attributes

<dl>
<dt><tt>axis</tt> : int (required)</dt>
<dd>Which axis to concat on</dd>
</dl>

#### Inputs (1 - &#8734;)

<dl>
<dt><tt>inputs</tt> (variadic) : T</dt>
<dd>List of tensors for concatenation</dd>
</dl>

#### Outputs

<dl>
<dt><tt>concat_result</tt> : T</dt>
<dd>Concatenated tensor</dd>
</dl>


### <a name="Constant"></a><a name="constant">Constant</a>

  A constant tensor.

#### Attributes

<dl>
<dt><tt>value</tt> : tensor (required)</dt>
<dd>The value for the elements of the output tensor.</dd>
</dl>

#### Inputs


#### Outputs

<dl>
<dt><tt>output</tt> : T</dt>
<dd>Output tensor containing the same value of the provided tensor.Usually used to store weights, bias, scalar etc , constant tensor</dd>
</dl>

### <a name="Conv"></a><a name="conv">Conv</a>

  The convolution operator consumes an input tensor and a filter, and
  computes the output.


#### Attributes

<dl>
<dt><tt>dilations</tt> : list of ints</dt>
<dd>dilation value along each axis of the filter. If not present, the dilation defaults to 1 along each axis.</dd>
<dt><tt>group</tt> : int</dt>
<dd>number of groups input channels and output channels are divided into, default is 1.</dd>
<dt><tt>kernel_shape</tt> : list of ints</dt>
<dd>The shape of the convolution kernel. If not present, should be inferred from input W.</dd>
<dt><tt>pads</tt> : string</dt>
<dd>"SAME" or "VALID"</dd>
<dt><tt>strides</tt> : list of ints</dt>
<dd>Stride along each axis. If not present, the stride defaults to 1 along each axis.</dd>
</dl>

#### Inputs (2 - 3)

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input data tensor from previous layer; has size (N x H x W x C), where N is the batch size, H and W are the height and width, and C is the number of channels. Note that this is for the 2D image. Otherwise the size is (N x D1 x D2 ... x Dn x C). Optionally, if dimension denotation is in effect, the operation expects input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_FEATURE, DATA_FEATURE ..., DATA_CHANNEL].</dd>
<dt><tt>W</tt> : T</dt>
<dd>The weight tensor that will be used in the convolutions; has size (M x kH x kW x C ), where C is the number of channels, and kH and kW are the height and width of the kernel, and M is the number of feature maps. For more than 2 dimensions, the kernel shape will be (M x k1 x k2 x ... x kn x C), where (k1 x k2 x ... kn) is the dimension of the kernel. </dd>
<dt><tt>B</tt> (optional) : T</dt>
<dd>Optional 1D bias to be added to the convolution, has size of M.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output data tensor that contains the result of the convolution. The output dimensions are functions of the kernel size, stride size, and pad type.</dd>
</dl>


### <a name="ConvTranspose"></a><a name="convtranspose">ConvTranspose</a>

注: 暂未遇到这个 OP, 暂定原有转换方式，后期根据具体情况再定协议进行修改。

  The convolution transpose operator consumes an input tensor and a filter,
  and computes the output.

  If the pads parameter is provided the shape of the output is calculated via the following equation:

    output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + kernel_shape[i] - pads[start_i] - pads[end_i]

  output_shape can also be explicitly specified in which case pads values are auto generated using these equations:

    total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + kernel_shape[i] - output_shape[i]
    If (auto_pads != SAME_UPPER): pads[start_i] = total_padding[i]/2; pads[end_i] = total_padding[i] - (total_padding[i]/2)
    Else: pads[start_i] = total_padding[i] - (total_padding[i]/2); pads[end_i] = (total_padding[i]/2).

#### Attributes

<dl>
<dt><tt>dilations</tt> : list of ints</dt>
<dd>dilation value along each axis of the filter. If not present, the dilation defaults to 1 along each axis.</dd>
<dt><tt>group</tt> : int</dt>
<dd>number of groups input channels and output channels are divided into, default is 1.</dd>
<dt><tt>kernel_shape</tt> : list of ints</dt>
<dd>The shape of the convolution kernel. If not present, should be inferred from input W.</dd>
<dt><tt>output_padding</tt> : list of ints</dt>
<dd>The zero-padding added to one side of the output. This is also called adjs/adjustment in some frameworks.</dd>
<dt><tt>output_shape</tt> : list of ints</dt>
<dd>The shape of the output can be explicitly set which will cause pads values to be auto generated. If output_shape is specified pads values are ignored. See doc for details for equations to generate pads</dd>
<dt><tt>pads</tt> : string</dt>
<dd>"SAME" or "VALID"</dd>
<dt><tt>strides</tt> : list of ints</dt>
<dd>Stride along each axis. If not present, the stride defaults to 1 along each axis.</dd>
</dl>

#### Inputs (2 - 3)
注：需修改

<dl>
<dt><tt>X</tt> : T</dt>

<dd>Input data tensor from previous layer; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the 2D image.Otherwise the size is (N x D1 x D2 ... x Dn)</dd>
<dt><tt>W</tt> : T</dt>
<dd>The weight tensor that will be used in the convolutions; has size (C x M x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel, and M is the number of feature maps. For more than 2 dimensions, the weight shape will be (C x M x k1 x k2 x ... x kn), where (k1 x k2 x ... x kn) is the dimension of the kernel</dd>
<dt><tt>B</tt> (optional) : T</dt>
<dd>Optional 1D bias to be added to the convolution, has size of C.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output data tensor that contains the result of the convolution. The output dimensions are functions of the kernel size, stride size, and pad lengths.</dd>
</dl>


### <a name="Elu"></a><a name="elu">Elu</a>

  Elu takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the function `f(x) = alpha * (exp(x) - 1.) for x <
  0`, `f(x) = x for x >= 0`., is applied to the tensor elementwise.

#### Attributes

<dl>
<dt><tt>alpha</tt> : float</dt>
<dd>Coefficient of ELU default to 1.0.</dd>
</dl>

#### Inputs

<dl>
<dt><tt>X</tt> : T</dt>
<dd>1D input tensor</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>1D input tensor</dd>
</dl>


### <a name="Flatten"></a><a name="flatten">Flatten</a>

  Flattens the input tensor into a 2D matrix. If input tensor has shape
  (d_0, d_1, ... d_n) then the output will have shape
  (d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn).

#### Attributes

<dl>
<dt><tt>axis</tt> : int</dt>
<dd>(Default to 1) Indicate up to which input dimensions (exclusive) should be flattened to the outer dimension of the output. The value for axis must be in the range [0, R], where R is the rank of the input tensor. When axis = 0, the shape of the output tensor is (1, (d_0 X d_1 ... d_n), where the shape of the input tensor is (d_0, d_1, ... d_n). </dd>
</dl>

#### Inputs

<dl>
<dt><tt>input</tt> : T</dt>
<dd>A tensor of rank >= axis.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>output</tt> : T</dt>
<dd>A 2D tensor with the contents of the input tensor, with input dimensions up to axis flattened to the outer dimension of the output and remaining input dimensions flattened into the inner dimension of the output.</dd>
</dl>


### <a name="Gemm"></a><a name="gemm">Gemm</a>

注：TF pb 中暂未遇到，Caffe 中有

  General Matrix multiplication:
  https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

  A' = transpose(A) if transA else A

  B' = transpose(B) if transB else B

  Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
  input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
  and output tensor Y has shape (M, N). A will be transposed before doing the
  computation if attribute transA is non-zero, same for B and transB.
  This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check [the doc](Broadcasting.md).

#### Attributes

<dl>
<dt><tt>alpha</tt> : float</dt>
<dd>Scalar multiplier for the product of input tensors A * B</dd>
<dt><tt>beta</tt> : float</dt>
<dd>Scalar multiplier for input tensor C</dd>
<dt><tt>transA</tt> : int</dt>
<dd>Whether A should be transposed</dd>
<dt><tt>transB</tt> : int</dt>
<dd>Whether B should be transposed</dd>
</dl>

#### Inputs

<dl>
<dt><tt>A</tt> : T</dt>
<dd>Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero.</dd>
<dt><tt>B</tt> : T</dt>
<dd>Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero.</dd>
<dt><tt>C</tt> : T</dt>
<dd>Input tensor C. The shape of C should be unidirectional broadcastable to (M, N).</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output tensor of shape (M, N).</dd>
</dl>

### <a name="GlobalAveragePool"></a><a name="globalaveragepool">GlobalAveragePool</a>

  GlobalAveragePool consumes an input tensor X and applies average pooling across the
   the values in the same channel. This is equivalent to AveragePool with kernel size
   equal to the spatial dimension of input tensor.


#### Inputs

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input data tensor from the previous operator; dimensions for image case are (N  x H x W x C), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x D1 x D2 ... Dn x C), where N is the batch size.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output data tensor from pooling across the input tensor. Dimensions will be N  x 1 x 1 x C</dd>
</dl>


### <a name="GlobalMaxPool"></a><a name="globalmaxpool">GlobalMaxPool</a>

  GlobalMaxPool consumes an input tensor X and applies max pooling across the
   the values in the same channel. This is equivalent to MaxPool with kernel size
   equal to the spatial dimension of input tensor.

#### Version

This version of the operator has been available since version 1 of the default ONNX operator set.

#### Inputs

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input data tensor from the previous operator; dimensions for image case are (N  x H x W x C), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N  x D1 x D2 ... Dn x C), where N is the batch size.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output data tensor from pooling across the input tensor. Dimensions will be N  x 1 x 1 x C</dd>
</dl>

### <a name="LeakyRelu"></a><a name="leakyrelu">LeakyRelu</a>

  LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
  output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
  `f(x) = x for x >= 0`, is applied to the data tensor elementwise.

#### Attributes

<dl>
<dt><tt>alpha</tt> : float</dt>
<dd>Coefficient of leakage default to 0.01.</dd>
</dl>

#### Inputs

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input tensor</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output tensor</dd>
</dl>


### <a name="MaxPool"></a><a name="maxpool">MaxPool</a>

  MaxPool consumes an input tensor X and applies max pooling across the
   the tensor according to kernel sizes, stride sizes, and pad lengths.
   max pooling consisting of computing the max on all values of a
   subset of the input tensor according to the kernel size and downsampling the
   data into the output tensor Y for further processing.

#### Attributes

<dl>
<dt><tt>kernel_shape</tt> : list of ints (required)</dt>
<dd>The size of the kernel along each axis.</dd>
<dt><tt>pads</tt> : string</dt>
<dd>"SAME" or "VALID"</dd>
<dt><tt>strides</tt> : list of ints</dt>
<dd>Stride along each axis. If not present, the stride defaults to 1 along each axis.</dd>
</dl>

#### Inputs

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input data tensor from the previous operator; dimensions for image case are (N  x H x W x C), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x D1 x D2 ... Dn x C), where N is the batch size. </dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output data tensor from average or max pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes. Floor value of the dimension is used</dd>
</dl>


### <a name="Mul"></a><a name="mul">Mul</a>

  Performs element-wise binary multiplication (with Numpy-style broadcasting not  support).

#### Inputs

<dl>
<dt><tt>A</tt> : T</dt>
<dd>First operand.</dd>
<dt><tt>B</tt> : T</dt>
<dd>Second operand.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>C</tt> : T</dt>
<dd>Result, has same element type as two inputs</dd>
</dl>



### <a name="PRelu"></a><a name="prelu">PRelu</a>

注: TF 中没有，caffe 中有

  PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
  output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
  `f(x) = x for x >= 0`., is applied to the data tensor elementwise.
  This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X).

#### Inputs

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input tensor</dd>
<dt><tt>slope</tt> : T</dt>
<dd>Slope tensor. The shape of slope can be smaller then first input X; if so, its shape must be unidirectional broadcastable to X</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output tensor (same size as X)</dd>
</dl>


### <a name="Pad"></a><a name="pad">Pad</a>

  Given `data` tensor, pads, mode, and value.

#### Attributes

<dl>
<dt><tt>mode</tt> : string</dt>
<dd>Three modes: constant(default), reflect, edge, now, we only support constant mode </dd>
<dt><tt>pads</tt> : list </dt>
<dd> same to tf.Pad , change to A list of type `int32`. </dd>
<dt><tt>value</tt> : float</dt>
<dd>One float, indicates the value to be filled, default is 0, now, we only support value = 0, like ZeroPadding.</dd>
</dl>

#### Inputs

<dl>
<dt><tt>data</tt> : T</dt>
<dd>Input tensor.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>output</tt> : T</dt>
<dd>Tensor after padding.</dd>
</dl>


### <a name="ReduceMean"></a><a name="reducemean">ReduceMean</a>

  Computes the mean of the input tensor's element along the provided axes. The resulted
  tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
  the resulted tensor have the reduced dimension pruned.

  The above behavior is similar to numpy, with the exception that numpy default keepdims to
  False instead of True.

#### Attributes

<dl>
<dt><tt>axes</tt> : list of ints</dt>
<dd>A list of integers, along which to reduce. The default is to reduce over all the dimensions of the input tensor.</dd>
<dt><tt>keepdims</tt> : int</dt>
<dd>Keep the reduced dimension or not, default 1 mean keep reduced dimension.</dd>
</dl>

#### Inputs

<dl>
<dt><tt>data</tt> : T</dt>
<dd>An input tensor.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>reduced</tt> : T</dt>
<dd>Reduced output tensor.</dd>
</dl>

### <a name="Relu"></a><a name="relu">Relu</a>

  Relu takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
  the tensor elementwise.


#### Inputs

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input tensor</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output tensor</dd>
</dl>


### <a name="Reshape"></a><a name="reshape">Reshape</a>

  Reshape the input tensor similar to numpy.reshape.
  First input is the data tensor, second input is a shape tensor which specifies the output shape. It outputs the reshaped tensor.
  At most one dimension of the new shape can be -1. In this case, the value is
  inferred from the size of the tensor and the remaining dimensions. A dimension
  could also be 0, in which case the actual dimension value is unchanged (i.e. taken
  from the input tensor).


#### Inputs

<dl>
<dt><tt>data</tt> : T</dt>
<dd>An input tensor.</dd>
<dt><tt>shape</tt> : tensor(int64)</dt>
<dd>Specified shape for output.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>reshaped</tt> : T</dt>
<dd>Reshaped data.</dd>
</dl>

### <a name="SpaceToDepth"></a><a name="spacetodepth">SpaceToDepth</a>

  SpaceToDepth rearranges blocks of spatial data into depth. More specifically,
  this op outputs a copy of the input tensor where values from the height and width dimensions
  are moved to the depth dimension.


#### Attributes

<dl>
<dt><tt>blocksize</tt> : int (required)</dt>
<dd>Blocks of [blocksize, blocksize] are moved.</dd>
</dl>

#### Inputs

<dl>
<dt><tt>input</tt> : T</dt>
<dd>Input tensor of [N,C,H,W], where N is the batch axis, C is the channel or depth, H is the height and W is the width.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>output</tt> : T</dt>
<dd>Output tensor of [N, C * blocksize * blocksize, H/blocksize, W/blocksize].</dd>
</dl>


### <a name="Sigmoid"></a><a name="sigmoid">Sigmoid</a>

  Sigmoid takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
  tensor elementwise.

#### Inputs

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input tensor</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output tensor</dd>
</dl>

### <a name="Slice"></a><a name="slice">Slice</a>

注： 需确定是否需要支持。

  Produces a slice of the input tensor along multiple axes. Similar to numpy:
  https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
  Slices uses `axes`, `starts` and `ends` attributes to specify the start and end
  dimension for each axis in the list of axes, it uses this information to
  slice the input `data` tensor. If a negative value is passed for any of the
  start or end indices, it represent number of elements before the end of that
  dimension. If the value passed to start or end is larger than the `n` (the
  number of elements in this dimension), it represents `n`. For slicing to the
  end of a dimension with unknown size, it is recommended to pass in `INT_MAX`.
  If `axes` are omitted, they are set to `[0, ..., ndim-1]`.
  Example 1:
    data = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
    axes = [0, 1]
    starts = [1, 0]
    ends = [2, 3]
    result = [
        [5, 6, 7],
    ]
  Example 2:
    data = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
    starts = [0, 1]
    ends = [-1, 1000]
    result = [
        [2, 3, 4],
    ]



#### Attributes

<dl>
<dt><tt>axes</tt> : list of ints</dt>
<dd>Axes that `starts` and `ends` apply to. It's optional. If not present, will be treated as [0, 1, ..., len(`starts`) - 1].</dd>
<dt><tt>ends</tt> : list of ints (required)</dt>
<dd>Ending indices (exclusive) of corresponding axis in `axes`</dd>
<dt><tt>starts</tt> : list of ints (required)</dt>
<dd>Starting indices of corresponding axis in `axes`</dd>
</dl>

#### Inputs

<dl>
<dt><tt>data</tt> : T</dt>
<dd>Tensor of data to extract slices from.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>output</tt> : T</dt>
<dd>Sliced data tensor.</dd>
</dl>


### <a name="Max"></a><a name="max">**Max**</a>

  Element-wise max of each of the input tensors (with Numpy-style broadcasting support).
  All inputs and outputs must have the same data type.
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

#### Inputs (1 - &#8734;)

<dl>
<dt><tt>data_0</tt> (variadic) : T</dt>
<dd>List of tensors for max.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>max</tt> : T</dt>
<dd>Output tensor.</dd>
</dl>

### <a name="MatMul"></a><a name="matmul">**MatMul**</a>

  Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html


#### Inputs

<dl>
<dt><tt>A</tt> : T</dt>
<dd>N-dimensional matrix A</dd>
<dt><tt>B</tt> : T</dt>
<dd>N-dimensional matrix B</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Matrix multiply results from A * B</dd>
</dl>


### <a name="Softmax"></a><a name="softmax">Softmax</a>

  The operator computes the softmax (normalized exponential) values for each layer in the batch
   of the given input. The input is a 2-D tensor (Tensor<float>) of size
  (batch_size x input_feature_dimensions). The output tensor has the same shape
  and contains the softmax values of the corresponding input.

  X does not need to explicitly be a 2D vector; rather, it will be
  coerced into one. For an arbitrary n-dimensional tensor
  X \in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] and k is
  the axis provided, then X will be coerced into a 2-dimensional tensor with
  dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default
  case where axis=1, this means the X tensor will be coerced into a 2D tensor
  of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the batch size.
  In this situation, we must have a_0 = N and a_1 * ... * a_{n-1} = D.
  Each of these dimensions must be matched correctly, or else the operator
  will throw errors.

#### Attributes

<dl>
<dt><tt>axis</tt> : int</dt>
<dd>(int) default to 1; describes the axis of the inputs when coerced to 2D; defaults to one because the 0th axis most likely describes the batch_size</dd>
</dl>

#### Inputs

<dl>
<dt><tt>input</tt> : T</dt>
<dd>The input tensor that's coerced into a 2D matrix of size (NxD) as described above.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>output</tt> : T</dt>
<dd>The output values with the same shape as input tensor.</dd>
</dl>



### <a name="Squeeze"></a><a name="squeeze">Squeeze</a>

  Remove single-dimensional entries from the shape of a tensor.
  Takes a  parameter `axes` with a list of axes to squeeze.
  If an axis is selected with shape entry not equal to one, an error is raised.

#### Attributes

<dl>
<dt><tt>axes</tt> : list of ints (required)</dt>
<dd>List of positive integers, indicate the dimensions to squeeze.</dd>
</dl>

#### Inputs

<dl>
<dt><tt>data</tt> : T</dt>
<dd>Tensors with at least max(dims) dimensions.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>squeezed</tt> : T</dt>
<dd>Reshaped tensor with same data as input.</dd>
</dl>


### <a name="Sum"></a><a name="sum">Sum</a>

  Element-wise sum of each of the input tensors. All inputs and outputs must
  have the same shape and data type.

#### Inputs (1 - &#8734;)

<dl>
<dt><tt>data_0</tt> (variadic) : T</dt>
<dd>List of tensors for Sum.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>sum</tt> : T</dt>
<dd>Output tensor. Same dimension as inputs.</dd>
</dl>


### <a name="Tanh"></a><a name="tanh">Tanh</a>

  Calculates the hyperbolic tangent of the given input tensor element-wise.

#### Inputs

<dl>
<dt><tt>input</tt> : T</dt>
<dd>Input tensor</dd>
</dl>

#### Outputs

<dl>
<dt><tt>output</tt> : T</dt>
<dd>The hyperbolic tangent values of the input tensor computed element-wise</dd>
</dl>



### <a name="Transpose"></a><a name="transpose">Transpose</a>

  Transpose the input tensor similar to numpy.transpose. For example, when
  perm=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape
  will be (2, 1, 3).


#### Attributes

<dl>
<dt><tt>perm</tt> : list of ints</dt>
<dd>A list of integers. By default, reverse the dimensions, otherwise permute the axes according to the values given.</dd>
</dl>

#### Inputs

<dl>
<dt><tt>data</tt> : T</dt>
<dd>An input tensor.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>transposed</tt> : T</dt>
<dd>Transposed output.</dd>
</dl>


### <sub>experimental</sub> <a name="Scale"></a><a name="scale">**Scale**</a>

  Scale takes one input data (Tensor<float>) and produces one output data
  (Tensor<float>) whose value is the input data tensor scaled element-wise.

#### Attributes

<dl>
<dt><tt>scale</tt> : float</dt>
<dd>(float, default 1.0) the scale to apply.</dd>
</dl>

#### Inputs

<dl>
<dt><tt>input</tt> : T</dt>
<dd>Input data to be scaled</dd>
</dl>

#### Outputs

<dl>
<dt><tt>output</tt> : T</dt>
<dd>Output data after scaling</dd>
</dl>

### <a name="UpSampling2D"></a><a name="UpSampling2D">UpSampling2D</a>

  Upsample the input tensor.
  Each dimension value of the output tensor is:
    output_dimension = floor(input_dimension * scale).

#### Attributes

<dl>
<dt><tt>mode</tt> : string</dt>
<dd>Two interpolation modes: nearest (default), and linear (including bilinear, trilinear, etc)</dd>
<dt><tt>size</tt> : list of ints (required)</dt>
<dd>The scale array along each dimension. It takes value greater than or equal to 1. The number of elements of 'scales' should be the same as the rank of input 'X'.</dd>
</dl>

#### Inputs

<dl>
<dt><tt>X</tt> : T</dt>
<dd>N-D tensor</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>N-D tensor after resizing</dd>
</dl>


### <a name="AddBias"></a><a name="AddBias">AddBias</a>

Performs element-wise binary addition (we do not support Numpy-style broadcasting).

#### Inputs

<dl>
<dt><tt>A</tt> : T</dt>
<dd>First operand.</dd>
<dt><tt>B</tt> : T</dt>
<dd>Second operand.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>C</tt> : T</dt>
<dd>Result, has same element type as two inputs</dd>
</dl>

### <a name="Relu6"></a><a name="Relu6">Relu6</a>

Computes Rectified Linear 6: `min(max(features, 0), 6)`.

Source: [Convolutional Deep Belief Networks on CIFAR-10. A.
Krizhevsky](http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf)

#### Inputs

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input tensor</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output tensor</dd>
</dl>


### <a name="DepthwiseConv2d"></a><a name="DepthwiseConv2d">DepthwiseConv2d</a>

Depthwise 2-D convolution.

  Given a 4D input tensor ('NHWC' or 'NCHW' data formats)
  and a filter tensor of shape
  `[filter_height, filter_width, in_channels, channel_multiplier]`
  containing `in_channels` convolutional filters of depth 1, `depthwise_conv2d`
  applies a different filter to each input channel (expanding from 1 channel
  to `channel_multiplier` channels for each), then concatenates the results
  together.  The output has `in_channels * channel_multiplier` channels.

  In detail,
  ```
  output[b, i, j, k * channel_multiplier + q] = sum_{di, dj}
       filter[di, dj, k, q] * input[b, strides[1] * i + rate[0] * di,
                                       strides[2] * j + rate[1] * dj, k]
  ```

  Must have `strides[0] = strides[3] = 1`.  For the most common case of the
  same horizontal and vertical strides, `strides = [1, stride, stride, 1]`.
  If any value in `rate` is greater than 1, we perform atrous depthwise
  convolution, in which case all values in the `strides` tensor must be equal
  to 1.

#### Attributes

<dl>
<dt><tt>dilations</tt> : list of ints</dt>
<dd>dilation value along each axis of the filter. If not present, the dilation defaults to 1 along each axis.</dd>
<dt><tt>group</tt> : int</dt>
<dd>number of groups input channels and output channels are divided into, default is 1.</dd>
<dt><tt>kernel_shape</tt> : list of ints</dt>
<dd>The shape of the convolution kernel. If not present, should be inferred from input W.</dd>
<dt><tt>pads</tt> : string</dt>
<dd>"SAME" or "VALID"</dd>
<dt><tt>strides</tt> : list of ints</dt>
<dd>Stride along each axis. If not present, the stride defaults to 1 along each axis.</dd>
</dl>

#### Inputs

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input data tensor from previous layer; has size (N x H x W x C), where N is the batch size, H and W are the height and width, and C is the number of channels. Note that this is for the 2D image. Otherwise the size is (N x D1 x D2 ... x Dn x C). Optionally, if dimension denotation is in effect, the operation expects input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_FEATURE, DATA_FEATURE ..., DATA_CHANNEL].</dd>
<dt><tt>W</tt> : T</dt>
<dd>The weight tensor that will be used in the convolutions; filter
[filter_height, filter_width, in_channels, channel_multiplier]</dd>
</dl>


#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output tensor</dd>
</dl>

### <a name="Dense"></a><a name="Dense">Dense</a>

Functional interface for the densely-connected layer.

  This layer implements the operation:
  `outputs = activation(inputs.kernel + bias)`
  Where `activation` is the activation function passed as the `activation`
  argument (if not `None`), `kernel` is a weights matrix created by the layer,
  and `bias` is a bias vector created by the layer
  (only if `use_bias` is `True`).

#### Attributes

<dl>
<dt><tt>dilations</tt> : list of ints</dt>
<dd>dilation value along each axis of the filter. If not present, the dilation defaults to 1 along each axis.</dd>
<dt><tt>group</tt> : int</dt>
<dd>number of groups input channels and output channels are divided into, default is 1.</dd>
<dt><tt>kernel_shape</tt> : list of ints</dt>
<dd>The shape of the convolution kernel. If not present, should be inferred from input W.</dd>
<dt><tt>pads</tt> : string</dt>
<dd>"SAME" or "VALID"</dd>
<dt><tt>strides</tt> : list of ints</dt>
<dd>Stride along each axis. If not present, the stride defaults to 1 along each axis.</dd>
</dl>

#### Inputs (2 - 3)

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input data tensor from previous layer; has size (N x H x W x C), where N is the batch size, H and W are the height and width, and C is the number of channels. Note that this is for the 2D image. Otherwise the size is (N x D1 x D2 ... x Dn x C). Optionally, if dimension denotation is in effect, the operation expects input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_FEATURE, DATA_FEATURE ..., DATA_CHANNEL].</dd>
<dt><tt>W</tt> : T</dt>
<dd>The weight tensor that will be used in the convolutions; has size (M x kH x kW x C ), where C is the number of channels, and kH and kW are the height and width of the kernel, and M is the number of feature maps. For more than 2 dimensions, the kernel shape will be (M x k1 x k2 x ... x kn x C), where (k1 x k2 x ... kn) is the dimension of the kernel. </dd>
<dt><tt>B</tt> (optional) : T</dt>
<dd>Optional 1D bias to be added to the convolution, has size of M.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output data tensor that contains the result of the convolution. The output dimensions are functions of the kernel size, stride size, and pad lengths.</dd>
</dl>
