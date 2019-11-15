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
''' loss implementation'''
import delta.compat as tf

from delta.utils.loss.base_loss import Loss
from delta.utils.loss.loss_utils import cross_entropy
from delta.utils.loss.loss_utils import mask_sequence_loss
from delta.utils.loss.loss_utils import ctc_lambda_loss
from delta.utils.loss.loss_utils import crf_log_likelihood
from delta.utils.loss.loss_utils import focal_loss

from delta.utils.register import registers


@registers.loss.register
class CrossEntropyLoss(Loss):
  ''' cross entropy loss for classfication and sequence classfication '''

  def __init__(self, config):
    super().__init__(config)
    self.smoothing = self.config['solver']['optimizer']['label_smoothing']

  #pylint: disable=too-many-arguments
  def call(self,
           logits=None,
           input_length=None,
           labels=None,
           label_length=None,
           **kwargs):

    loss = cross_entropy(
        logits=logits,
        input_length=input_length,
        labels=labels,
        label_length=label_length,
        smoothing=self.smoothing)
    return loss


@registers.loss.register
class DistillationLoss(Loss):
  ''' Distilling the Knowledge in a Neural Network, arXiv:1503.02531 '''

  def __init__(self, config):
    super().__init__(config)
    self.smoothing = self.config['solver']['optimizer']['label_smoothing']
    self.temperature = self.config['solver']['distilling']['temperature']
    self.alpha = self.config['solver']['distilling']['alpha']

    assert self.alpha >= 0.0, "alpha : {}".format(self.alpha)
    assert self.alpha <= 1.0, "alpha : {}".format(self.alpha)
    assert self.temperature >= 1, "temperature : {}".format(self.temperature)
    self.T = tf.convert_to_tensor(self.temperature, dtype=tf.float32)  #pylint: disable=invalid-name

  #pylint: disable=too-many-arguments
  def call(self,
           logits=None,
           input_length=None,
           labels=None,
           label_length=None,
           **kwargs):

    assert "soft_lables" in kwargs
    soft_labels = kwargs["soft_labels"]

    loss_standard = cross_entropy(
        logits=logits,
        input_length=input_length,
        labels=labels,
        label_length=label_length,
        smoothing=self.smoothing)
    loss_soft = cross_entropy(
        logits=logits / self.T,
        input_length=input_length,
        labels=soft_labels,
        label_length=label_length,
        smoothing=self.smoothing)
    # Since the magnitudes of the gradients produced by the soft targets
    # scale as 1/T2 , it is important to multiply them by T2 when using
    # both hard and soft targets
    total_loss = self.alpha * tf.square(
        self.T) * loss_soft + (1 - self.alpha) * loss_standard

    return total_loss


@registers.loss.register
class CTCLoss(Loss):
  ''' ctc loss '''

  def __init__(self, config):  #pylint: disable=useless-super-delegation
    super().__init__(config)

  #pylint: disable=too-many-arguments
  def call(self,
           logits=None,
           input_length=None,
           labels=None,
           label_length=None,
           **kwargs):

    blank_index = kwargs.get('blank_index', 0)
    return ctc_lambda_loss(
        logits=logits,
        input_length=input_length,
        labels=labels,
        label_length=label_length,
        blank_index=blank_index)


@registers.loss.register
class CrfLoss(Loss):
  '''crf loss for sequence labeling'''

  def __init__(self, config):
    super().__init__(config)

  # pylint: disable=too-many-arguments
  def call(self,
           logits=None,
           input_length=None,
           labels=None,
           label_length=None,
           **kwargs):
    assert "model" in kwargs
    model = kwargs["model"]
    tags_scores = tf.reshape(
        logits, [-1, model.max_len, model.seq_num_classes], name="scores")
    loss, _ = crf_log_likelihood(tags_scores, labels, input_length,
                                 model.transitions)

    return loss


@registers.loss.register
class SequenceCrossEntropyLoss(Loss):
  ''' cross entropy loss for sequence to sequence '''

  def __init__(self, config):
    super().__init__(config)

  #pylint: disable=too-many-arguments
  def call(self,
           logits=None,
           input_length=None,
           labels=None,
           label_length=None,
           **kwargs):

    loss = mask_sequence_loss(logits, labels, input_length, label_length)
    return loss


@registers.loss.register
class FocalLoss(Loss):

  def __init__(self, config):
    super().__init__(config)

    class_num = self._config['data']['task']['classes']['num']
    if 'alpha' in self._config['data']['task']['classes']:
      self.alpha = self._config['data']['task']['classes']['alpha']
      assert len(self.alpha) == class_num, 'alpha len is not equal to class_num'
      self.alpha = tf.constant(self.alpha)
    else:
      self.alpha = tf.ones([class_num])

    self.gamma = 2
    if 'gamma' in self._config['solver']['optimizer']:
      self.gamma = self._config['solver']['optimizer']['gamma']
    assert self.gamma >= 0, 'gamma must greater than or equal to zero'

  def call(self,
           logits=None,
           input_length=None,
           labels=None,
           label_length=None,
           **kwargs):

    del input_length
    del label_length

    return focal_loss(
        logits=logits, labels=labels, gamma=self.gamma, name='focal_loss')
