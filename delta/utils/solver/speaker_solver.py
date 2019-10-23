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
''' Solver for speaker classification. '''
from absl import logging
import librosa
import delta.compat as tf

from delta.utils.solver.estimator_solver import EstimatorSolver
from delta.utils.register import registers


@registers.solver.register
class SpeakerSolver(EstimatorSolver):
  ''' For training speaker recognition models. '''

  def process_config(self, config):
    data_conf = config['data']

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
    # with batch_size
    taskconf = self.config['data']['task']
    shape = [None] + taskconf['audio']['feature_shape']
    logging.debug('serving input shape:{}'.format(shape))

    return tf.estimator.export.build_raw_serving_input_receiver_fn(
        features={
            'inputs':
                tf.placeholder(name="inputs", shape=shape, dtype=tf.float32),
        },
        default_batch_size=None,
    )
