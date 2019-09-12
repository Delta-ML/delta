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
''' Callback utilities.'''

from absl import logging

import delta.compat as tf
#pylint: disable=import-error
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
#pylint: disable=no-name-in-module
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import dataset_ops

from delta.utils.decode import py_ctc
from delta.utils.decode import tf_ctc
from delta.utils import metrics as metrics_lib


#pylint: disable=too-few-public-methods
class TokenErrMetricCallBack(Callback):
  '''Callback to compute specific metric and logs during train and eval'''

  def __init__(self, func, eval_ds, eval_task, decoder_type):
    self.func = func
    self.eval_task = eval_task
    self.eval_ds = eval_ds
    self.next_batch_gen = None
    self.decoder_type = decoder_type

  #pylint: disable=dangerous-default-value
  def on_epoch_end(self, epoch, logs={}):
    '''computing token error'''

    cur_session = K.get_session()
    target_seq_list, predict_seq_list = [], []

    is_py_sequence = True
    if isinstance(self.eval_ds, (dataset_ops.DatasetV2, dataset_ops.DatasetV1)):
      eval_gen = self.eval_ds.make_one_shot_iterator()
      self.next_batch_gen = eval_gen.get_next()[0]
      is_py_sequence = False
    elif isinstance(self.eval_ds,
                    (iterator_ops.IteratorV2, iterator_ops.Iterator)):
      self.next_batch_gen = self.ds.get_next()[0]
      is_py_sequence = False

    for index in range(len(self.eval_task)):
      batch_data = None
      if is_py_sequence:
        batch_data = self.eval_ds[index][0]
      else:
        batch_data = cur_session.run(self.next_batch_gen)
      batch_input = batch_data['inputs']
      batch_target = batch_data['targets'].tolist()
      batch_predict = self.func(batch_input)[0]

      if self.decoder_type == 'argmax':
        predict_seq_list += py_ctc.ctc_greedy_decode(
            batch_predict, 0, unique=True)
      else:
        sequence_lens = [len(pre_sequence) for pre_sequence in batch_predict]
        batch_decoder, _ = tf_ctc.ctc_beam_search_decode(
            tf.constant(batch_predict),
            tf.constant(sequence_lens),
            beam_width=3,
            top_paths=3)
        predict_seq_list += cur_session.run(batch_decoder)[0].tolist()
      target_seq_list += batch_target

    val_token_errors = metrics_lib.token_error(
        predict_seq_list=predict_seq_list,
        target_seq_list=target_seq_list,
        eos_id=0)
    logs['val_token_err'] = val_token_errors

    if 'val_loss' in logs:
      logging.info("Epoch {}: on eval, val_loss is {}.".format(
          epoch + 1, logs['val_loss']))
    logging.info("Epoch {}: on eval, token_err is {}.".format(
        epoch + 1, val_token_errors))
    logging.info("Epoch {}: loss on train is {}".format(epoch + 1,
                                                        logs['loss']))


class ParallelModelCheckpoint(ModelCheckpoint):
  '''Callback to save multi_gpu_model'''

  #pylint: disable=too-many-arguments
  def __init__(self,
               model,
               filepath,
               monitor='val_loss',
               verbose=0,
               save_best_only=False,
               save_weights_only=False,
               mode='auto',
               save_freq='epoch',
               load_weights_on_restart=False,
               period=1):
    self.model_to_save = model
    super().__init__(
        filepath=filepath,
        monitor=monitor,
        verbose=verbose,
        save_best_only=save_best_only,
        save_weights_only=save_weights_only,
        mode=mode,
        save_freq=save_freq,
        load_weights_on_restart=load_weights_on_restart,
        period=period)

  #pylint: disable=unused-argument
  def set_model(self, model):
    '''set the model to saved'''
    super().set_model(self.model_to_save)
