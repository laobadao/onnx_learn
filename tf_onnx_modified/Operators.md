## Operator Schemas
*在 onnx 基础上的修改版。*

* ai.onnx (modified)
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
  * <a href="#Identity">Identity</a>  
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
  * <sub>modified</sub> <a href="#Add_">Add_</a>
  * <sub>modified</sub> <a href="#UpSampling2D">UpSampling2D</a>  
  * <sub>modified</sub> <a href="#Relu6">Relu6</a>
  * <a href="#LeakyRelu">LeakyRelu</a>
  * <sub>modified</sub> <a href="#LeakyRelu">LeakyRelu</a>
  * <sub>modified</sub> <a href="#DepthwiseConv2d">DepthwiseConv2d</a>
  * <sub>modified</sub> <a href="#Dense">Dense</a>


## ai.onnx (modified)
### <a name="Abs"></a><a name="abs">**Abs**</a>

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


### <a name="Add"></a><a name="add">**Add**</a>

  Performs element-wise binary addition (with Numpy-style broadcasting support).

  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

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


### <a name="AveragePool"></a><a name="averagepool">**AveragePool**</a>

  AveragePool consumes an input tensor X and applies average pooling across the
   the tensor according to kernel sizes, stride sizes, and pad lengths.
   average pooling consisting of computing the average on all values of a
   subset of the input tensor according to the kernel size and downsampling the
   data into the output tensor Y for further processing. The output spatial shape will be following:
   ```
   output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)

   * pad_shape[i] is sum of pads along axis i
   ```

   `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
   ```
   VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
   SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
   ```
   And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
   ```
   pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
   ```
   The output of each pooling window is divided by the number of elements (exclude pad when attribute count_include_pad is zero).

#### Attributes

<dl>
<dt><tt>auto_pad</tt> : string</dt>
<dd>auto_pad must be either SAME_UPPER, SAME_LOWER or VALID. Where SAME_UPPER or SAME_LOWER mean pad the input so that the output size match the input.In case of odd number add the extra padding at the end for SAME_UPPER and at the beginning for SAME_LOWER. VALID mean no padding. DEPRECATION NOTE: auto_pad is only intended to support legacy uses, and for framework authors, one is explicitly encouraged to use explicit padding specified in the pads attribute.</dd>
<dt><tt>count_include_pad</tt> : int</dt>
<dd>Whether include pad pixels when calculating values for the edges.</dd>
<dt><tt>kernel_shape</tt> : list of ints (required)</dt>
<dd>The size of the kernel along each axis.</dd>
<dt><tt>pads</tt> : list of ints</dt>
<dd>Padding for the beginning and ending along each axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each axis.</dd>
<dt><tt>strides</tt> : list of ints</dt>
<dd>Stride along each axis. If not present, the stride defaults to 1 along each axis.</dd>
</dl>

#### Inputs

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size. Optionally, if dimension denotation is in effect, the operation expects the input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output data tensor from average or max pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes. Floor value of the dimension is used</dd>
</dl>


### <a name="BatchNormalization"></a><a name="batchnormalization">**BatchNormalization**</a>

  Carries out batch normalization as described in the paper
  https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
  there are multiple cases for the number of outputs, which we list below:

  Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
  Output case #2: Y (test mode)
      This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

#### Attributes

<dl>
<dt><tt>epsilon</tt> : float</dt>
<dd>The epsilon value to use to avoid division by zero, default is 1e-5f.</dd>
<dt><tt>momentum</tt> : float</dt>
<dd>Factor used in computing the running mean and variance.e.g., running_mean = running_mean * momentum + mean * (1 - momentum), default is 0.9f.</dd>
<dt><tt>spatial</tt> : int</dt>
<dd>If true, compute the mean and variance across all spatial elements If false, compute the mean and variance across per feature.Default is 1.</dd>
</dl>

#### Inputs

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.</dd>
<dt><tt>scale</tt> : T</dt>
<dd>The scale as a 1-dimensional tensor of size C to be applied to the output.</dd>
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
<dd>The running mean after the BatchNormalization operator.</dd>
<dt><tt>var</tt> (optional) : T</dt>
<dd>The running variance after the BatchNormalization operator.</dd>
<dt><tt>saved_mean</tt> (optional) : T</dt>
<dd>Saved mean used during training to speed up gradient computation.</dd>
<dt><tt>saved_var</tt> (optional) : T</dt>
<dd>Saved variance used during training to speed up gradient computation.</dd>
</dl>


### <a name="Concat"></a><a name="concat">**Concat**</a>

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


### <a name="Constant"></a><a name="constant">**Constant**</a>

  A constant tensor.

#### Version

This version of the operator has been available since version 1 of the default ONNX operator set.

#### Attributes

<dl>
<dt><tt>value</tt> : tensor (required)</dt>
<dd>The value for the elements of the output tensor.</dd>
</dl>

#### Inputs


#### Outputs

<dl>
<dt><tt>output</tt> : T</dt>
<dd>Output tensor containing the same value of the provided tensor.</dd>
</dl>

### <a name="Conv"></a><a name="conv">**Conv**</a>

  The convolution operator consumes an input tensor and a filter, and
  computes the output.


#### Attributes

<dl>
<dt><tt>auto_pad</tt> : string</dt>
<dd>auto_pad must be either SAME_UPPER, SAME_LOWER or VALID. Where SAME_UPPER or SAME_LOWER mean pad the input so that the output size match the input.In case of odd number add the extra padding at the end for SAME_UPPER and at the beginning for SAME_LOWER. VALID mean no padding. DEPRECATION NOTE: auto_pad is only intended to support legacy uses, and for framework authors, one is explicitly encouraged to use explicit padding specified in the pads attribute.</dd>
<dt><tt>dilations</tt> : list of ints</dt>
<dd>dilation value along each axis of the filter. If not present, the dilation defaults to 1 along each axis.</dd>
<dt><tt>group</tt> : int</dt>
<dd>number of groups input channels and output channels are divided into, default is 1.</dd>
<dt><tt>kernel_shape</tt> : list of ints</dt>
<dd>The shape of the convolution kernel. If not present, should be inferred from input W.</dd>
<dt><tt>pads</tt> : list of ints</dt>
<dd>Padding for the beginning and ending along each axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each axis.</dd>
<dt><tt>strides</tt> : list of ints</dt>
<dd>Stride along each axis. If not present, the stride defaults to 1 along each axis.</dd>
</dl>

#### Inputs (2 - 3)

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input data tensor from previous layer; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the 2D image. Otherwise the size is (N x C x D1 x D2 ... x Dn). Optionally, if dimension denotation is in effect, the operation expects input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].</dd>
<dt><tt>W</tt> : T</dt>
<dd>The weight tensor that will be used in the convolutions; has size (M x C x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel, and M is the number of feature maps. For more than 2 dimensions, the kernel shape will be (M x C x k1 x k2 x ... x kn), where (k1 x k2 x ... kn) is the dimension of the kernel. Optionally, if dimension denotation is in effect, the operation expects the weight tensor to arrive with the dimension denotation of [FILTER_IN_CHANNEL, FILTER_OUT_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL ...].</dd>
<dt><tt>B</tt> (optional) : T</dt>
<dd>Optional 1D bias to be added to the convolution, has size of M.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output data tensor that contains the result of the convolution. The output dimensions are functions of the kernel size, stride size, and pad lengths.</dd>
</dl>


### <a name="ConvTranspose"></a><a name="convtranspose">**ConvTranspose**</a>

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
<dt><tt>auto_pad</tt> : string</dt>
<dd>auto_pad must be either SAME_UPPER, SAME_LOWER or VALID. Where SAME_UPPER or SAME_LOWER mean pad the input so that the output size match the input.In case of odd number add the extra padding at the end for SAME_UPPER and at the beginning for SAME_LOWER. VALID mean no padding. DEPRECATION NOTE: auto_pad is only intended to support legacy uses, and for framework authors, one is explicitly encouraged to use explicit padding specified in the pads attribute.</dd>
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
<dt><tt>pads</tt> : list of ints</dt>
<dd>Padding for the beginning and ending along each axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each axis.</dd>
<dt><tt>strides</tt> : list of ints</dt>
<dd>Stride along each axis. If not present, the stride defaults to 1 along each axis.</dd>
</dl>

#### Inputs (2 - 3)

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


### <a name="Elu"></a><a name="elu">**Elu**</a>

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


### <a name="Flatten"></a><a name="flatten">**Flatten**</a>

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


### <a name="Gemm"></a><a name="gemm">**Gemm**</a>

  General Matrix multiplication:
  https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

  A' = transpose(A) if transA else A

  B' = transpose(B) if transB else B

  Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
  input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
  and output tensor Y has shape (M, N). A will be transposed before doing the
  computation if attribute transA is non-zero, same for B and transB.
  This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check [the doc](Broadcasting.md).

#### Version

This version of the operator has been available since version 7 of the default ONNX operator set.

Other versions of this operator: <a href="Changelog.md#Gemm-1">Gemm-1</a>, <a href="Changelog.md#Gemm-6">Gemm-6</a>

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

### <a name="GlobalAveragePool"></a><a name="globalaveragepool">**GlobalAveragePool**</a>

  GlobalAveragePool consumes an input tensor X and applies average pooling across the
   the values in the same channel. This is equivalent to AveragePool with kernel size
   equal to the spatial dimension of input tensor.


#### Inputs

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output data tensor from pooling across the input tensor. Dimensions will be N x C x 1 x 1</dd>
</dl>


### <a name="GlobalMaxPool"></a><a name="globalmaxpool">**GlobalMaxPool**</a>

  GlobalMaxPool consumes an input tensor X and applies max pooling across the
   the values in the same channel. This is equivalent to MaxPool with kernel size
   equal to the spatial dimension of input tensor.

#### Version

This version of the operator has been available since version 1 of the default ONNX operator set.

#### Inputs

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output data tensor from pooling across the input tensor. Dimensions will be N x C x 1 x 1</dd>
</dl>

### <a name="LeakyRelu"></a><a name="leakyrelu">**LeakyRelu**</a>

  LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
  output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
  `f(x) = x for x >= 0`, is applied to the data tensor elementwise.

#### Version

This version of the operator has been available since version 6 of the default ONNX operator set.

Other versions of this operator: <a href="Changelog.md#LeakyRelu-1">LeakyRelu-1</a>

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


### <a name="MaxPool"></a><a name="maxpool">**MaxPool**</a>

  MaxPool consumes an input tensor X and applies max pooling across the
   the tensor according to kernel sizes, stride sizes, and pad lengths.
   max pooling consisting of computing the max on all values of a
   subset of the input tensor according to the kernel size and downsampling the
   data into the output tensor Y for further processing. The output spatial shape will be following:
   ```
   output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)

   * pad_shape[i] is sum of pads along axis i
   ```

   `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
   ```
   VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
   SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
   ```
   And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
   ```
   pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
   ```
   The output of each pooling window is maximum number of elements exclude pad.

#### Attributes

<dl>
<dt><tt>auto_pad</tt> : string</dt>
<dd>auto_pad must be either SAME_UPPER, SAME_LOWER or VALID. Where SAME_UPPER or SAME_LOWER mean pad the input so that the output size match the input.In case of odd number add the extra padding at the end for SAME_UPPER and at the beginning for SAME_LOWER. VALID mean no padding. DEPRECATION NOTE: auto_pad is only intended to support legacy uses, and for framework authors, one is explicitly encouraged to use explicit padding specified in the pads attribute.</dd>
<dt><tt>kernel_shape</tt> : list of ints (required)</dt>
<dd>The size of the kernel along each axis.</dd>
<dt><tt>pads</tt> : list of ints</dt>
<dd>Padding for the beginning and ending along each axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each axis.</dd>
<dt><tt>strides</tt> : list of ints</dt>
<dd>Stride along each axis. If not present, the stride defaults to 1 along each axis.</dd>
</dl>

#### Inputs

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size. Optionally, if dimension denotation is in effect, the operation expects the input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output data tensor from average or max pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes. Floor value of the dimension is used</dd>
</dl>


### <a name="Mul"></a><a name="mul">**Mul**</a>

  Performs element-wise binary multiplication (with Numpy-style broadcasting support).

  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).


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



### <a name="PRelu"></a><a name="prelu">**PRelu**</a>

  PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
  output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
  `f(x) = x for x >= 0`., is applied to the data tensor elementwise.
  This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X); for more details please check [the doc](Broadcasting.md).

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


### <a name="Pad"></a><a name="pad">**Pad**</a>

  Given `data` tensor, pads, mode, and value.
  Example:
    Insert 0 pads to the beginning of the second dimension.
    data = [
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7],
    ]
    pads = [0, 2, 0, 0]
    output = [
        [
            [0.0, 0.0, 1.0, 1.2],
            [0.0, 0.0, 2.3, 3.4],
            [0.0, 0.0, 4.5, 5.7],
        ],
    ]

#### Attributes

<dl>
<dt><tt>mode</tt> : string</dt>
<dd>Three modes: constant(default), reflect, edge</dd>
<dt><tt>pads</tt> : list of ints (required)</dt>
<dd>List of integers indicating the number of padding elements to add or remove (if negative) at the beginning and end of each axis. For 2D it is the number of pixels. `pads` rank should be double of the input's rank. `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`.</dd>
<dt><tt>value</tt> : float</dt>
<dd>One float, indicates the value to be filled, default is 0</dd>
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


### <a name="ReduceMean"></a><a name="reducemean">**ReduceMean**</a>

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

### <a name="Relu"></a><a name="relu">**Relu**</a>

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


### <a name="Reshape"></a><a name="reshape">**Reshape**</a>

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


### <a name="Sigmoid"></a><a name="sigmoid">**Sigmoid**</a>

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

### <a name="Slice"></a><a name="slice">**Slice**</a>

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
<dd>Ending indices (exclusive) of corresponding axis in axes`</dd>
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


### <a name="Softmax"></a><a name="softmax">**Softmax**</a>

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



### <a name="Squeeze"></a><a name="squeeze">**Squeeze**</a>

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


### <a name="Sum"></a><a name="sum">**Sum**</a>

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


### <a name="Tanh"></a><a name="tanh">**Tanh**</a>

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



### <a name="Transpose"></a><a name="transpose">**Transpose**</a>

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


### <a name="Upsample"></a><a name="upsample">**Upsample**</a>

  Upsample the input tensor.
  Each dimension value of the output tensor is:
    output_dimension = floor(input_dimension * scale).

#### Attributes

<dl>
<dt><tt>mode</tt> : string</dt>
<dd>Two interpolation modes: nearest (default), and linear (including bilinear, trilinear, etc)</dd>
<dt><tt>scales</tt> : list of floats (required)</dt>
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
