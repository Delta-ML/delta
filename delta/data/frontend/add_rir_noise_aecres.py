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
"""This model adds noise/rir to signal."""

import delta.compat as tf
from delta.utils.hparam import HParams
from core.ops import py_x_ops
from delta.data.frontend.base_frontend import BaseFrontend


class Add_rir_noise_aecres(BaseFrontend):
  """
  Add a random signal-to-noise ratio noise or impulse response to clean speech.
  """

  def __init__(self, config: dict):
    super().__init__(config)

  @classmethod
  def params(cls, config=None):
    """
        Set params.
        :param config: contains nine optional parameters:
            --sample_rate				  : Sample frequency of waveform data. (int, default = 16000)
            --if_add_rir          : If true, add rir to audio data. (bool, default = False)
            --rir_filelist        : FileList path of rir.(string, default = 'rirlist.scp')
            --if_add_noise        : If true, add random noise to audio data. (bool, default = False)
            --snr_min             : Minimum SNR adds to signal. (float, default = 0)
            --snr_max             : Maximum SNR adds to signal. (float, default = 30)
            --noise_filelist      : FileList path of noise.(string, default = 'noiselist.scp')
            --if_add_aecres       : If true, add aecres to audio data. (bool, default = False)
            --aecres_filelist     : FileList path of aecres.(string, default = 'aecreslist.scp')
        :return: An object of class HParams, which is a set of hyperparameters as name-value pairs.
        """

    sample_rate = 16000
    if_add_rir = False
    rir_filelist = 'rirlist.scp'
    if_add_noise = False
    noise_filelist = 'noiselist.scp'
    snr_min = 0
    snr_max = 30
    if_add_aecres = False
    aecres_filelist = 'aecreslist.scp'

    hparams = HParams(cls=cls)
    hparams.add_hparam('sample_rate', sample_rate)
    hparams.add_hparam('if_add_rir', if_add_rir)
    hparams.add_hparam('if_add_noise', if_add_noise)
    hparams.add_hparam('rir_filelist', rir_filelist)
    hparams.add_hparam('noise_filelist', noise_filelist)
    hparams.add_hparam('snr_min', snr_min)
    hparams.add_hparam('snr_max', snr_max)
    hparams.add_hparam('if_add_aecres', if_add_aecres)
    hparams.add_hparam('aecres_filelist', aecres_filelist)

    if config is not None:
      hparams.override_from_dict(config)

    return hparams

  def call(self, audio_data, sample_rate=None):
    """
        Caculate power spectrum or log power spectrum of audio data.
        :param audio_data: the audio signal from which to compute spectrum.
                          Should be an (1, N) tensor.
        :param sample_rate: [option]the samplerate of the signal we working with,
                           default is 16kHz.
        :return: A float tensor of size N containing add-noise audio.
        """

    p = self.config
    with tf.name_scope('add_rir_noise_aecres'):
      if sample_rate == None:
        sample_rate = tf.constant(p.sample_rate, dtype=tf.int32)

      assert_op = tf.assert_equal(
          tf.constant(p.sample_rate), tf.cast(sample_rate, dtype=tf.int32))
      with tf.control_dependencies([assert_op]):
        sample_rate = tf.cast(sample_rate, dtype=float)
        add_rir_noise_aecres_out = py_x_ops.add_rir_noise_aecres(
            audio_data,
            sample_rate,
            if_add_rir=p.if_add_rir,
            rir_filelist=p.rir_filelist,
            if_add_noise=p.if_add_noise,
            snr_min=p.snr_min,
            snr_max=p.snr_max,
            noise_filelist=p.noise_filelist,
            if_add_aecres=p.if_add_aecres,
            aecres_filelist=p.aecres_filelist)

        return tf.squeeze(add_rir_noise_aecres_out)
