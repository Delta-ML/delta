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
''' A sequential ASR task. '''
import numpy as np
import delta.compat as tf
from absl import logging

from delta import utils
from delta.data.utils import espnet_utils

from delta.utils.register import registers
from delta.data.task.base_speech_task import SpeechTask

# pylint: disable=consider-using-enumerate
# pylint: disable=invalid-name


def _make_example(uttids, feats, ilens, targets, olens):
  features = {
      'uttids':
          uttids,
      'inputs':
          tf.expand_dims(feats, axis=-1) if not isinstance(feats, np.ndarray)
          else np.expand_dims(feats, axis=-1),
      'input_length':
          ilens,
      'targets':
          targets,
      'target_length':
          olens
  }
  labels = {
      'ctc':
          tf.ones(tf.shape(feats)[0])
          if not isinstance(feats, np.ndarray) else np.ones(feats.shape[0])
  }  # dummy data for dummy loss function
  return features, labels


@registers.task.register  #pylint: disable=too-many-instance-attributes
class AsrSeqTask(SpeechTask, tf.keras.utils.Sequence):
  ''' ASR Task '''

  def __init__(self, config, mode):
    super().__init__(config, mode)
    self.dummy = config['data']['task']['dummy']
    self.batch_mode = config['data']['task']['batch_mode']
    self.batch_size = config['solver']['optimizer']['batch_size']
    self._shuffle_buffer_size = config['data']['task']['shuffle_buffer_size']
    self._need_shuffle = config['data']['task'][
        'need_shuffle'] and mode == utils.TRAIN
    # get batches form data path
    if self.dummy:
      self._feat_shape = [40]
      logging.info("Dummy data: feat {}".format(self.feat_shape))
      self._vocab_size = 100
    else:
      data_metas = espnet_utils.get_batches(self.config, mode)
      self.batches = data_metas['data']
      self.n_utts = data_metas['n_utts']
      logging.info("utts: {}".format(self.n_utts))
      # [nframe, feat_shape, ...]
      self._feat_shape = self.batches[0][0][1]['input'][0]['shape'][1:]
      # [tgt_len, vocab_size]
      self._vocab_size = self.batches[0][0][1]['output'][0]['shape'][1]
    logging.info('#input feat shape: ' + str(self.feat_shape))
    logging.info('#output dims: ' + str(self.vocab_size))

    self._converter = espnet_utils.ASRConverter(self.config)
    self.on_epoch_end()

  @property
  def converter(self):
    ''' return self._converter which is espnet_utils.ASRConverter '''
    return self._converter

  @property
  def feat_shape(self):
    ''' Feature shape. '''
    assert isinstance(self._feat_shape, (list))
    return self._feat_shape

  @property
  def vocab_size(self):
    ''' Vocabulary size. '''
    return self._vocab_size

  @property
  def steps_per_epoch(self):
    ''' Steps per epoch. '''
    if self.dummy:
      step = 1
      logging.info("Dummy data: step {}".format(step))
      return step

    steps = None
    if self.batch_mode:
      steps = len(self.batches)
    else:
      batch_size = self._config['data']['task']['batch']['batch_size']
      steps = int(self.n_utts / batch_size)
    return steps

  def generate_feat(self, paths):
    pass

  def generate_cmvn(self, paths):
    pass

  def __len__(self):
    ''' Denotes the number of batches per epoch'''
    return self.steps_per_epoch

  def on_epoch_end(self):
    '''shuffle data after each epoch'''
    self.batch_num = self.steps_per_epoch
    self.indexes = np.arange(self.batch_num)
    if self._need_shuffle:
      np.random.shuffle(self.indexes)

  def __getitem__(self, batch_index):
    ''' Generates a batch of correctly shaped X and Y data
    :param batch_index: index of the batch to generate
    :return: batch of (x, y)
    '''

    assert self.batch_mode
    batch_index_after_shuffle = self.indexes[batch_index]
    batch = self.batches[batch_index_after_shuffle]

    uttids, feats, ilens, targets, olens = self._process_batch(batch)
    return _make_example(uttids, feats, ilens, targets, olens)

  #pylint: disable=too-many-locals
  def _process_batch(self, batch):
    srcs, ilens, tgts, olens, uttid_list = self.converter(batch)

    imax = max(ilens)
    omax = max(olens)
    batch_feat = []
    batch_target = []
    for i in range(len(srcs)):
      #pad feat
      ipad_len = imax - srcs[i].shape[0]
      feat = np.pad(
          srcs[i],
          pad_width=((0, ipad_len), (0, 0)),
          mode='constant',
          constant_values=0)

      #pad target
      opad_len = omax - tgts[i].shape[0]
      target = np.pad(
          tgts[i], pad_width=(0, opad_len), mode='constant', constant_values=0)

      batch_feat.append(feat)
      batch_target.append(target)

    batch_uttid = np.array(uttid_list)
    batch_feat = np.stack(batch_feat).astype(np.float32)
    batch_target = np.stack(batch_target).astype(np.int64)
    ilens = np.array(ilens).astype(np.int64)
    olens = np.array(olens).astype(np.int64)

    return batch_uttid, batch_feat, ilens, batch_target, olens

  def generate_data(self):  #pylint: disable=too-many-locals
    '''
        :return: feat, feat_len, target, terget_len
        '''
    if self.batch_mode:
      for batch in self.batches:
        batch_uttid, batch_feat, ilens, batch_target, olens = self._process_batch(
            batch)
        yield batch_uttid, batch_feat, ilens, batch_target, olens
    else:
      for batch in self.batches:
        srcs, ilens, tgts, olens, uttid_list = self.converter(batch)
        for i in range(len(srcs)):
          yield uttid_list[i], srcs[i], ilens[i], tgts[i], olens[i]

  def feature_spec(self, batch_size_):  # pylint: disable=arguments-differ
    '''
        uttid: []
        feat: [feat_shape]
        src_len: []
        label: [None]
        tgt_len: []
        '''
    values = None
    batch_size = None
    time = None
    if self.dummy:
      batch_size = batch_size_
      time = 10
      logging.info("Dummy data: batch size {} time {}".format(batch_size, time))

    types = (tf.string, tf.float32, tf.int32, tf.int32, tf.int32)
    if self.batch_mode or self.dummy:
      # batch of examples
      shapes = (
          #uttid
          tf.TensorShape([batch_size]),
          # input
          tf.TensorShape([batch_size, time, *self.feat_shape]),
          # input len
          tf.TensorShape([batch_size]),
          # output
          tf.TensorShape([batch_size, time]),
          # output len
          tf.TensorShape([batch_size]),
      )
    else:
      # one example
      shapes = (
          #uttid
          tf.TensorShape([]),
          # input
          tf.TensorShape([time, *self.feat_shape]),
          # input len
          tf.TensorShape([]),
          # output
          tf.TensorShape([time]),
          # output len
          tf.TensorShape([]),
      )
    if self.dummy:
      values = ("uttid_1", 1, 2, 3, 4)
      logging.info("Dummy data: shapes {}".format(shapes))
      logging.info("Dummy data: types {}".format(types))
      logging.info("Dummy data: values {}".format(values))
    return types, shapes, values

  def preprocess_batch(self, batch):
    return batch

  def dataset(self, mode, batch_size, epoch):  # pylint: disable=arguments-differ
    if batch_size != self.batch_size:
      logging.warning("dataset: batch_size not equal to config: {} {}".format(
          batch_size, self.batch_size))

    types, shapes, values = self.feature_spec(batch_size)
    logging.debug('dtypes: {} shapes: {} values: {}'.format(
        types, shapes, values))

    if self.dummy:
      logging.info("Dummy data: dataset")
      dss = []
      for i in range(len(shapes)):
        dss.append(
            utils.generate_synthetic_data(
                input_shape=shapes[i],
                input_value=values[i],
                input_dtype=types[i],
                nepoch=epoch))
      ds = tf.data.Dataset.zip(tuple(dss))
    else:
      del values
      ds = tf.data.Dataset.from_generator(
          generator=lambda: self.generate_data(),  # pylint: disable=unnecessary-lambda
          output_types=types,
          output_shapes=shapes)

      if mode == utils.TRAIN:
        if self._need_shuffle:
          ds = ds.shuffle(self._shuffle_buffer_size, seed=None)
        ds = ds.repeat(count=epoch)

      if not self.batch_mode:
        ds = ds.padded_batch(
            batch_size,
            padded_shapes=shapes,
            padding_values=None,
            drop_remainder=True if mode == utils.TRAIN else False)  #pylint: disable=simplifiable-if-expression

    ds = ds.map(_make_example)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

  def batch_input_shape(self):
    ''' batch input TensorShape '''
    feature, labels = self.__getitem__(0)

    feature_shape, label_shape = {}, {}
    for feature_key, feature_val in feature.items():
      feature_shape[feature_key] = tf.TensorShape((None,) +
                                                  feature_val.shape[1:])

    for label_key, label_val in labels.items():
      label_shape[label_key] = tf.TensorShape((None,) + label_val.shape[1:])

    return feature_shape, label_shape
