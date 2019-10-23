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
''' misc.py unittest'''
import numpy as np
import delta.compat as tf

from delta.utils import misc


class MiscTest(tf.test.TestCase):
  ''' misc unittest'''

  def setUp(self):
    super().setUp()
    '''setup'''
    self.length = [3, 5, 2]
    self.mask_true = np.array([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0],
    ])

  def tearDown(self):
    '''tear down'''

  def test_len_to_mask(self):
    ''' len to mask unittest'''
    with self.cached_session(use_gpu=False, force_gpu=False):
      mask = misc.len_to_mask(self.length, dtype=tf.int32)
      self.assertAllEqual(mask.eval(), self.mask_true)

  def test_len_to_padding(self):
    ''' len to padding unittest'''
    with self.cached_session(use_gpu=False, force_gpu=False):
      padding = misc.len_to_padding(self.length, dtype=tf.int32)
      self.assertAllEqual(padding.eval(), 1 - self.mask_true)

  def test_gpu_device_names(self):
    ''' gpu device names unittest'''
    with self.cached_session(use_gpu=False, force_gpu=False):
      devices, ngpus = misc.gpu_device_names()
      self.assertListEqual(devices, [])
      self.assertEqual(ngpus, 0)

  def test_per_device_batch_size(self):
    ''' per device batch size unittest'''
    batch_size, ngpus = 32, 2
    batch_per_dev = misc.per_device_batch_size(batch_size, ngpus)
    self.assertEqual(batch_per_dev, 16)

    batch_size, ngpus = 32, 1
    batch_per_dev = misc.per_device_batch_size(batch_size, ngpus)
    self.assertEqual(batch_per_dev, 32)

    with self.assertRaises(ValueError):
      batch_size, ngpus = 32, 3
      batch_per_dev = misc.per_device_batch_size(batch_size, ngpus)

  def test_generate_synthetic_data(self):
    ''' generate sythetic data unittest'''
    input_shape = tf.TensorShape([2, 3])
    input_value = 1
    input_dtype = tf.float32
    label_shape = tf.TensorShape([2])
    label_value = 2
    label_dtype = tf.int32
    nepoch = 2

    data_set = misc.generate_synthetic_data(input_shape, input_value,
                                            input_dtype, label_shape,
                                            label_value, label_dtype, nepoch)

    iterator = data_set.make_one_shot_iterator()

    with self.cached_session(use_gpu=False, force_gpu=False):
      data, label = iterator.get_next()
      self.assertAllEqual(data.eval(),
                          np.ones(shape=input_shape, dtype=np.float32))
      self.assertAllEqual(label.eval(),
                          2 * np.ones(shape=label_shape, dtype=np.float32))

      with self.assertRaises(tf.errors.OutOfRangeError):
        data.eval()


if __name__ == '__main__':
  tf.test.main()
