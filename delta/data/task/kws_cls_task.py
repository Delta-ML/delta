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
''' kws task '''
# data format see: docs/data/kws.md
import struct
import numpy as np
import delta.compat as tf

from delta import utils
from delta.utils.register import registers
from delta.data.task.base_speech_task import SpeechTask
from delta.data.utils.htk_reader_lib import HtkReaderIO


@registers.task.register
class KwsClsTask(SpeechTask):
  ''' kws task '''

  #pylint: disable=too-many-instance-attributes
  def __init__(self, config, mode):
    super().__init__(config, mode)
    self.epoch = 0
    self.num = 0
    self.reader = HtkReaderIO()

    self.window_len = config['data']['task']['audio']['window_len']
    self.window_shift = config['data']['task']['audio']['window_shift']
    self.cmvn_path = config['data']['task']['audio']['cmvn_path']

    self.left_context = config['data']['task']['audio']['left_context']
    self.right_context = config['data']['task']['audio']['right_context']
    self.delta_order = config['data']['task']['audio']['delta_order']
    self.delta_wind = config['data']['task']['audio']['delta_wind']
    self.splice_frame = config['data']['task']['audio']['splice_frame']

    feat_dim = config['data']['task']['audio']['feat_dim']
    if self.splice_frame:
      feat_dim = config['data']['task']['audio']['feat_dim'] * (
          self.left_context + 1 + self.right_context)
    self.final_feat_dim = feat_dim * (self.delta_order + 1)

    if mode == utils.TRAIN:
      self.lines = open(config['data']['train']['paths']).readlines()
    else:
      self.lines = open(config['data']['eval']['paths']).readlines()

  def generate_feat(self, paths):
    ''' generate feature'''

  def generate_cmvn(self, paths):
    ''' generate cmvn '''

  #pylint: disable=too-many-locals
  def generate_data(self):
    '''
    train.list file:
      /path/to/10w.42.feat
      /path/to/10w.42.label
      ./10w.42.desc
      /path/to/train.7.feat
      /path/to/train.7.label
      ./train.7.desc
    '''

    for i in range(0, len(self.lines), 3):
      fp_feat = open(self.lines[i].strip(), 'rb')
      buff = open(self.lines[i + 1].strip(), 'rb').read()
      # label is 0 ~ 8,
      # one label per frame
      label_arr = struct.unpack('%di' % (len(buff) / 4), buff)  # 570485
      #desc_lines = open(self.lines[i + 2].strip()).readlines()[1:]

      # read file header, frame_bytes is 160 Bytes, 40 dimensions
      num_frames, _, frame_bytes, _ = struct.unpack('!%di%dh' % (2, 2),
                                                    fp_feat.read(12))
      del num_frames
      buff = fp_feat.read()  # file body
      fp_feat.close()

      # ! means converting Big-Endian to Little-Endian
      feat_all = struct.unpack('!%df' % (len(buff) / 4), buff)
      feat_matrix = np.array(feat_all).reshape(
          (-1, int(frame_bytes / 4)))  # (570485, 40) (frame_num, feat_dim)

      #num, bad = 0, 0
      length = feat_matrix.shape[0] - self.window_len  #  281508
      for j in range(0, length, self.window_shift):
        label_t = np.unique(label_arr[j:j + self.window_len])
        if -1 in label_t:
          # reduce the ratio of negative samples
          continue
        if len(label_t) > 2 and len(label_t) < 8:
          continue

        feat = feat_matrix[j:j + self.window_len]
        _, feat = self.reader.add_delta(feat, self.delta_order, self.delta_wind)
        # cmvn is 120 lines, each line has mean and variance
        _, feat = self.reader.normalization_feat_by_mean_variance(
            feat, self.cmvn_path)
        if self.splice_frame:
          _, feat = self.reader.splice_frames(feat, self.left_context,
                                              self.left_context)
        if set(label_t).issuperset(range(0, 8)):
          # including keyword
          label = 1
        else:
          label = 0

        yield feat, label

  def feature_spec(self):
    ''' data meta'''
    output_shapes = (tf.TensorShape([self.window_len,
                                     self.final_feat_dim]), tf.TensorShape([]))
    output_types = (tf.float32, tf.int32)
    return output_shapes, output_types

  def preprocess_batch(self, batch):
    ''' preprocess of data'''
    return batch

  #pylint: disable=arguments-differ
  def dataset(self, mode, batch_size, epoch):
    ''' make tf dataset'''
    shapes, types = self.feature_spec()
    ds = tf.data.Dataset.from_generator(  #pylint: disable=invalid-name
        generator=lambda: self.generate_data(),  #pylint: disable=unnecessary-lambda
        output_types=types,
        output_shapes=shapes,
    )

    if mode == utils.TRAIN:
      ds = ds.apply(  #pylint: disable=invalid-name
          tf.data.experimental.shuffle_and_repeat(
              buffer_size=batch_size, count=epoch, seed=None))

    def make_sample(feat, label):
      return {"inputs": feat, "labels": label}, label

    return ds.apply(
        tf.data.experimental.map_and_batch(
            make_sample, batch_size,
            drop_remainder=False)).prefetch(tf.data.experimental.AUTOTUNE)
