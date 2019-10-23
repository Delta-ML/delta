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
''' CMVN unittest'''
import os
import numpy as np
import delta.compat as tf

from delta import utils


class CmvnTest(tf.test.TestCase):
  ''' CMVN unittest Class'''

  def setUp(self):
    super().setUp()
    ''' setup '''

  def tearDown(self):
    ''' tear down '''

  def testCreateCmvnStatis(self):  #pylint: disable=invalid-name
    ''' test creat_cmvn_statics '''
    feat_size = 40
    delta_deltas = True

    sums, square, count = utils.create_cmvn_statis(feat_size, delta_deltas)

    self.assertAllEqual(sums.shape, [1, feat_size, 3])
    self.assertAllEqual(square.shape, [1, feat_size, 3])
    self.assertAllEqual(count, 0)

    delta_deltas = False
    sums, square, count = utils.create_cmvn_statis(feat_size, delta_deltas)
    self.assertAllEqual(sums.shape, [1, feat_size, 1])
    self.assertAllEqual(square.shape, [1, feat_size, 1])
    self.assertAllEqual(count, 0)

  def testUpdateCmvnStatis(self):  #pylint: disable=invalid-name
    ''' test update cmvn statics '''
    np.random.seed(12)
    feat_size = 40
    delta_deltas = True
    shape = [2, 10, feat_size, 3 if delta_deltas else 1]

    sums, square, count = utils.create_cmvn_statis(feat_size, delta_deltas)

    feat = np.random.randn(*shape)
    sums_true = np.expand_dims(np.sum(feat, axis=(0, 1)), axis=0)
    square_true = np.expand_dims(np.sum(np.square(feat), axis=(0, 1)), axis=0)
    count_true = np.prod(shape[:2])

    sums, square, count = utils.update_cmvn_statis(feat, sums, square, count)

    self.assertAllEqual(sums, sums_true)
    self.assertAllEqual(square, square_true)
    self.assertAllEqual(count, count_true)

  def testComputeCmvn(self):  #pylint: disable=invalid-name
    ''' test compute cmvn '''
    np.random.seed(12)
    feat_size = 40
    delta_deltas = True
    shape = [2, 10, feat_size, 3 if delta_deltas else 1]

    sums, square, count = utils.create_cmvn_statis(feat_size, delta_deltas)

    feat = np.random.randn(*shape)
    feat = feat.astype(np.float32)
    sums, square, count = utils.update_cmvn_statis(feat, sums, square, count)
    mean, var = utils.compute_cmvn(sums, square, count)
    mean_true, var_true = np.mean(feat, axis=(0, 1)), np.var(feat, axis=(0, 1))

    self.assertAllEqual(mean.shape, [1] + shape[2:])
    self.assertAllClose(np.squeeze(mean, axis=0), mean_true)
    self.assertAllClose(np.squeeze(var, axis=0), var_true)

  def testLoadCmvn(self):  #pylint: disable=invalid-name
    ''' test load cmvn '''
    np.random.seed(12)
    temp_dir = self.get_temp_dir()
    temp_file = os.path.join(temp_dir, 'cmvn.npy')

    feat_size = 40
    delta_deltas = True
    shape = [1, feat_size, 3 if delta_deltas else 1]
    mean = np.random.randn(*shape)
    var = np.random.randn(*shape)
    mean, var = mean.astype(np.float32), var.astype(np.float32)
    with tf.gfile.Open(temp_file, 'w') as f:  #pylint: disable=invalid-name
      np.save(f, (mean, var))

    mean_true = np.expand_dims(mean, axis=0)
    var_true = np.expand_dims(var, axis=0)

    with self.cached_session(use_gpu=False, force_gpu=False):
      mean, var = utils.load_cmvn(temp_file)
      self.assertAllClose(mean.eval(), mean_true)
      self.assertAllClose(var.eval(), var_true)

  def testApplyCmvn(self):  #pylint: disable=invalid-name
    ''' test apply cmvn '''
    np.random.seed(12)
    tf.set_random_seed(12)

    feat_size = 40
    delta_deltas = True

    feat_shape = [2, 10, feat_size, 3 if delta_deltas else 1]
    feat = np.random.randn(*feat_shape)
    feat = feat.astype(np.float32)

    feat = tf.constant(feat)
    mean = feat / 2
    var = feat / 3

    eps = 1e-9
    feat_out = utils.apply_cmvn(feat, mean, var, epsilon=eps)
    feat_true = (feat - mean) * tf.rsqrt(var + eps)
    with self.cached_session(use_gpu=False, force_gpu=False):
      self.assertAllClose(feat_out.eval(), feat_true.eval())

  def testApplyLocalCmvn(self):  #pylint: disable=invalid-name
    ''' test apply_local_cmvn() '''
    np.random.seed(12)
    tf.set_random_seed(12)

    feat_size = 40
    delta_deltas = True

    feat_shape = [2, 10, feat_size, 3 if delta_deltas else 1]
    feat = np.random.randn(*feat_shape)
    feat = feat.astype(np.float32)

    mean = np.mean(feat, axis=1, keepdims=True)
    var = np.var(feat, axis=1, keepdims=True)
    eps = 1e-9
    feat_true = (feat - mean) / np.sqrt(var + eps)

    feat = tf.constant(feat)

    feat_out = utils.apply_local_cmvn(feat, epsilon=eps)
    with self.cached_session(use_gpu=False, force_gpu=False):
      self.assertAllClose(feat_out.eval(), feat_true)


if __name__ == '__main__':
  tf.test.main()
