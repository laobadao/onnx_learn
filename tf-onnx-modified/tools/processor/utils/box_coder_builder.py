# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A function to build an object detection box coder from configuration."""
from tools.processor.utils import faster_rcnn_box_coder


def build(box_coder_config):
    """Builds a box coder object based on the box coder config.

    Args:
      box_coder_config: A box_coder.proto object containing the config for the
        desired box coder.

    Returns:
      BoxCoder based on the config.

    Raises:
      ValueError: On empty box coder proto.
    """
    y_scale = 10.0
    x_scale = 10.0
    height_scale = 5.0
    width_scale = 5.0

    if box_coder_config == 'faster_rcnn_box_coder':
        return faster_rcnn_box_coder.FasterRcnnBoxCoder(scale_factors=[y_scale, x_scale, height_scale, width_scale])
    raise ValueError('Empty box coder.')
