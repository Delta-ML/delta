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
''' global CMVN functions '''
import numpy as np
import delta.compat as tf
#pylint: disable=no-name-in-module
from tensorflow.python.keras import backend as keras_backend


def create_cmvn_statis(feature_size, add_delta_deltas=True):
  ''' init sums, squares and cout of feature statistic '''
  sums = np.zeros([1, feature_size, 3 if add_delta_deltas else 1],
                  dtype=np.float64)
  square = np.zeros([1, feature_size, 3 if add_delta_deltas else 1],
                    dtype=np.float64)
  count = 0.0
  return sums, square, count


def update_cmvn_statis(feat, sums, square, count, axis=(0, 1)):
  ''' aggregate CMVN statistic '''
  # feat shape [ batch, frames, feat, channle]
  assert feat.ndim == 4
  sums += np.expand_dims(np.sum(feat, axis=axis), axis=0)
  square += np.expand_dims(np.sum(np.square(feat), axis=axis), axis=0)
  count += np.prod(feat.shape[:len(axis)])
  return sums, square, count


def compute_cmvn(sums, square, count):
  ''' compute global feature mean and variance
     vars = E(x^2) - (E(x))^2
  '''
  mean = sums / count
  var = (square / count) - np.square(mean)
  return mean, var


def load_cmvn(path):
  ''' load mean and variance from cmvn.npy,
      then convert to TF Tensor
  '''
  # [1, nbins, nchannels]
  mean, variance = np.load(path)
  # [1, 1, nbins, nchannels]
  mean = np.expand_dims(mean, axis=0)
  variance = np.expand_dims(variance, axis=0)
  mean = tf.convert_to_tensor(mean, dtype=tf.float32, name='cmvn_mean')
  variance = tf.convert_to_tensor(
      variance, dtype=tf.float32, name='cmvn_variance')
  return mean, variance


def apply_cmvn(feats, mean, variance, epsilon=1e-9):
  ''' TF: apply CMVN on feature'''
  return (feats - mean) * tf.rsqrt(variance + epsilon)


def apply_local_cmvn(feats, epsilon=1e-9):
  ''' feats: (NHWC) '''
  mean = tf.expand_dims(keras_backend.mean(feats, axis=1), axis=1)
  var = tf.expand_dims(keras_backend.var(feats, axis=1), axis=1)
  feats = (feats - mean) * tf.rsqrt(var + epsilon)
  return feats
