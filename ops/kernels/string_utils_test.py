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
''' string utils op unittest'''
import delta.compat as tf

from delta.layers.ops import py_x_ops


class StringUtilsOpTest(tf.test.TestCase):
  ''' string utils test'''

  def setUp(self):
    super().setUp()

  def tearDown(self):
    ''' tear down '''

  def test_lower(self):
    ''' test lower string'''
    with self.cached_session(use_gpu=False, force_gpu=False):
      output = py_x_ops.str_lower("Hello WORLD").eval()
      self.assertEqual(b'hello world', output)
      output = py_x_ops.str_lower(["Hello WORLD", "ABC XYZ"]).eval()
      self.assertAllEqual([b'hello world', b'abc xyz'], output)


if __name__ == '__main__':
  tf.test.main()
