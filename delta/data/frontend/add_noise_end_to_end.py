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
"""This model adds noise/rir to signal and writes it to file."""

import delta.compat as tf
from delta.utils.hparam import HParams
from delta.data.frontend.read_wav import ReadWav
from delta.data.frontend.add_rir_noise_aecres import Add_rir_noise_aecres
from delta.data.frontend.write_wav import WriteWav
from delta.data.frontend.base_frontend import BaseFrontend


class AddNoiseEndToEnd(BaseFrontend):
  """
  Add a random signal-to-noise ratio noise or impulse response to clean speech, and
  write it to wavfile.
  """

  def __init__(self, config: dict):
    super().__init__(config)
    self.add_noise = Add_rir_noise_aecres(config)
    self.read_wav = ReadWav(config)
    self.write_wav = WriteWav(config)

  @classmethod
  def params(cls, config=None):
    """
        Set params.
        :param config: contains ten optional parameters:
            --sample_rate				  : Sample frequency of waveform data. (int, default = 16000)
            --if_add_rir          : If true, add rir to audio data. (bool, default = False)
            --rir_filelist        : FileList path of rir.(string, default = 'rirlist.scp')
            --if_add_noise        : If true, add random noise to audio data. (bool, default = False)
            --snr_min             : Minimum SNR adds to signal. (float, default = 0)
            --snr_max             : Maximum SNR adds to signal. (float, default = 30)
            --noise_filelist      : FileList path of noise.(string, default = 'noiselist.scp')
            --if_add_aecres       : If true, add aecres to audio data. (bool, default = False)
            --aecres_filelist     : FileList path of aecres.(string, default = 'aecreslist.scp')
            --speed               : Speed of sample channels wanted. (float, default=1.0)
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
    audio_channels = 1
    speed = 1.0

    hparams = HParams(cls=cls)
    hparams.add_hparam('sample_rate', sample_rate)
    hparams.add_hparam('speed', speed)
    hparams.add_hparam('if_add_rir', if_add_rir)
    hparams.add_hparam('if_add_noise', if_add_noise)
    hparams.add_hparam('rir_filelist', rir_filelist)
    hparams.add_hparam('noise_filelist', noise_filelist)
    hparams.add_hparam('snr_min', snr_min)
    hparams.add_hparam('snr_max', snr_max)
    hparams.add_hparam('if_add_aecres', if_add_aecres)
    hparams.add_hparam('aecres_filelist', aecres_filelist)
    hparams.add_hparam('audio_channels', audio_channels)

    if config is not None:
      hparams.override_from_dict(config)

    return hparams

  def call(self, in_wavfile, out_wavfile):
    """
        Read a clean wav return a noisy wav.
        :param in_wavfile: clean wavfile path.
        :param out_wavfile: noisy wavfile path.
        :return: write wav opration.
        """

    with tf.name_scope('add_noise_end_to_end'):
      input_data, sample_rate = self.read_wav(in_wavfile)
      noisy_data = self.add_noise(input_data, sample_rate) / 32768
      write_op = self.write_wav(out_wavfile, noisy_data, sample_rate)

    return write_op
