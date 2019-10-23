# Copyright (C) 2017 Beijing Didi Infinity Technology and Development Co.,Ltd.
# All rights reserved.
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

import delta.compat as tf
import sys
import os
'''
python3 tflite_convert.py saved_model_dir
'''

os.environ['CUDA_VISIBLE_DEVICES'] = ''

converter = tf.contrib.lite.TFLiteConverter.from_saved_model(
    saved_model_dir=sys.argv[1], input_shapes={'inputs': [5, 2000, 40, 3]})
converter.dump_graphviz_dir = './lite_dump'
'''
converter.inference_type=tf.contrib.lite.constants.QUANTIZED_UINT8
converter.quantized_input_stats={'inputs': (127, 1.0/128)}
converter.default_ranges_stats=(0, 6)
'''

tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
