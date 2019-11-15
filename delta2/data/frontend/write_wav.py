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

from delta.utils.hparam import HParams
from delta.data.frontend.base_frontend import BaseFrontend


class WriteWav(BaseFrontend):

  def __init__(self, config: dict):
    super().__init__(config)

  @classmethod
  def params(cls, config=None):
    """
      Set params.
       :param config: contains one optional parameters:sample_rate(float, default=16000.0).
       :return: An object of class HParams, which is a set of hyperparameters as name-value pairs.
       """

    sample_rate = 16000.0

    hparams = HParams(cls=cls)
    hparams.add_hparam('sample_rate', sample_rate)

    if config is not None:
      hparams.override_from_dict(config)

    return hparams

  def call(self, filename, audio_data, sample_rate=None):
    """
    Write wav using audio_data[tensor].
    :param filename: filepath of wav.
    :param audio_data: a tensor containing data of a wav.
    :param sample_rate: [option]the samplerate of the signal we working with, default is 16kHz.
    :return: write wav opration.
    """
    p = self.config
    filename = tf.constant(filename)

    if sample_rate == None:
      sample_rate = tf.constant(p.sample_rate, dtype=tf.int32)

    assert_op = tf.assert_equal(
        tf.constant(p.sample_rate), tf.cast(sample_rate, dtype=float))
    with tf.control_dependencies([assert_op]):
      audio_data = tf.cast(audio_data, dtype=tf.float32)
      contents = tf.audio.encode_wav(
          tf.expand_dims(audio_data, 1), tf.cast(sample_rate, dtype=tf.int32))
      w = tf.io.write_file(filename, contents)

    return w
