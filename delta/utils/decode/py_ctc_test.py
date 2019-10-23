# Copyright (C) 2017 Beijing Didi Infinity Technology and Development Co.,Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
''' ctc python decoder test '''

import delta.compat as tf
from delta.utils.decode import py_ctc


class PyCtcTest(tf.test.TestCase):
  ''' ctc python decode unittest'''

  def setUp(self):
    super().setUp()
    ''' setup '''
    self.model_output = [[[0.1, 0.3, 0.5, 0.1], [0.1, 0.3, 0.5, 0.1],
                          [0.5, 0.1, 0.3, 0.1]],
                         [[0.1, 0.2, 0.3, 0.4], [0.3, 0.4, 0.2, 0.1]]]

  def tearDown(self):
    ''' tear down '''

  def test_ctc_greedy_decode(self):
    ''' ctc greedy decode unittest'''
    decode_result_list = py_ctc.ctc_greedy_decode(
        self.model_output, 0, unique=False)
    self.assertEqual(decode_result_list, [[2, 2], [3, 1]])

    decode_result_list = py_ctc.ctc_greedy_decode(
        self.model_output, 0, unique=True)
    self.assertEqual(decode_result_list, [[2], [3, 1]])


if __name__ == '__main__':
  tf.test.main()
