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
''' Speech Emotion Solver based on EstimatorSolver'''
import librosa
from absl import logging

#pylint: disable=no-name-in-module
import delta.compat as tf
from tensorflow.python.keras.utils import losses_utils

from delta import utils
from delta.utils.solver.estimator_solver import EstimatorSolver
from delta.utils.solver.asr_solver import AsrSolver
from delta.utils.register import registers


@registers.solver.register
class EmotionSolver(EstimatorSolver):
  ''' Speech Emotion Solver base on Estimator'''

  #pylint: disable=useless-super-delegation
  def __init__(self, config):
    super().__init__(config)

  def process_config(self, config):
    ''' preprocess config '''
    data_conf = config['data']
    class_vocab = data_conf['task']['classes']['vocab']
    assert len(class_vocab) == data_conf['task']['classes']['num']

    # add revere_vocab, positive_id
    reverse_vocab = {val: key for key, val in class_vocab.items()}
    data_conf['task']['classes']['reverse_vocab'] = reverse_vocab

    # binary class
    pos_id = config['solver']['metrics']['pos_label']
    data_conf['task']['classes']['positive_id'] = pos_id
    data_conf['task']['classes']['positive'] = reverse_vocab[pos_id]

    # add feature shape, withoud batch_size
    if data_conf['task']['suffix'] == '.npy':
      input_channels = 3 if data_conf['task']['audio']['add_delta_deltas'] else 1
      nframe = librosa.time_to_frames(
          data_conf['task']['audio']['clip_size'],
          sr=data_conf['task']['audio']['sr'],
          hop_length=data_conf['task']['audio']['winstep'] *
          data_conf['task']['audio']['sr'])
      feature_shape = [
          nframe, data_conf['task']['audio']['feature_size'], input_channels
      ]
    else:
      feature_shape = [
          data_conf['task']['audio']['sr'] *
          data_conf['task']['audio']['clip_size']
      ]
    data_conf['task']['audio']['feature_shape'] = feature_shape
    return config

  def create_serving_input_receiver_fn(self):
    ''' infer input pipeline '''
    # with batch_size
    taskconf = self.config['data']['task']
    shape = [None] + taskconf['audio']['feature_shape']
    logging.debug('serving input shape:{}'.format(shape))

    #pylint: disable=no-member
    return tf.estimator.export.build_raw_serving_input_receiver_fn(
        features={
            'inputs':
                tf.placeholder(name="inputs", shape=shape, dtype=tf.float32),
            'texts':
                tf.placeholder(
                    name="texts",
                    shape=(None, taskconf['text']['max_text_len']),
                    dtype=tf.int32)
        },
        default_batch_size=None,
    )


@registers.solver.register
class EmoKerasSolver(AsrSolver):
  ''' emotion keras solver '''

  def __init__(self, config):
    super().__init__(config)
    self.batch_input_shape = None
    self._label_smoothing = config['solver']['optimizer']['label_smoothing']

  @property
  def model(self):
    ''' keras Model '''
    return self.raw_model

  def input_fn(self, mode):
    ''' input function for tf.data.Dataset'''
    super().input_fn(mode)
    assert self.task
    self.batch_input_shape = self.task.batch_input_shape()
    return None, self.task

  def input_data(self, mode):
    ''' get input data '''
    _, task = self.input_fn(mode)
    assert self.task
    return None, task

  def get_loss(self):
    ''' keras losses  '''
    return tf.keras.losses.CategoricalCrossentropy(
        from_logits=True,
        label_smoothing=self._label_smoothing,
        reduction=losses_utils.ReductionV2.AUTO)

  def eval(self):
    ''' evaluation '''
    # must first construct input data, then build model
    eval_ds, eval_task = self.input_data(mode=utils.EVAL)
    self.model_fn(mode=utils.EVAL)
    assert self._built

    callbacks = []

    self.active_model.evaluate_generator(
        eval_task,
        steps=len(eval_task),
        verbose=1,
        callbacks=callbacks,
        max_queue_size=20,
        workers=1,
        use_multiprocessing=False)

    logging.info("Eval End.")

  def infer(self, yield_single_examples=False):
    ''' inference '''
    logging.fatal("Not Implemented")

  def export_model(self):
    logging.fatal("Not Implemented")
