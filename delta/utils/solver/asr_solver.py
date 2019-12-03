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
''' asr sovler based on Solver'''

from pathlib import Path
from datetime import datetime

from absl import logging
import delta.compat as tf

#pylint: disable=import-error
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.experimental import export_saved_model

from delta import utils
from delta.utils.decode import py_ctc
from delta.utils import metrics as metrics_lib
from delta.utils.solver.keras_base_solver import KerasBaseSolver
from delta.utils.register import registers
from delta.utils.solver.utils.callbacks import TokenErrMetricCallBack
from delta.utils.decode.tf_ctc import ctc_greedy_decode


#pylint: disable=too-many-instance-attributes,too-many-public-methods
@registers.solver.register
class AsrSolver(KerasBaseSolver):
  ''' asr keras solver'''

  def __init__(self, config):
    super().__init__(config)

  def input_fn(self, mode):
    ''' input function for tf.data.Dataset'''
    super().input_fn(mode)
    assert self.task
    self.batch_input_shape = self.task.batch_input_shape()
    batch_size = self.config['solver']['optimizer']['batch_size']
    num_epoch = self.config['solver']['optimizer']['epochs']
    return self.task.input_fn(mode, batch_size, num_epoch), self.task

  def input_data(self, mode):
    ''' input data '''
    input_fn, _task = self.input_fn(mode)
    ds_ = input_fn()
    #iterator = ds_.make_one_shot_iterator()
    #return iterator, task
    return ds_, _task

  #pylint: disable=no-self-use
  def get_loss(self):
    ''' dummy ctc loss, since ctc is implemented as a kearas layer '''
    loss = {'ctc': lambda y_true, y_pred: tf.reduce_mean(y_pred)}
    return loss

  def get_metric_callbacks(self, eval_gen, eval_task, monitor_used,
                           decoder_type):
    ''' metric_specific callbacks'''
    callbacks = []

    if monitor_used == 'val_token_err':
      metric_func = self.get_metric_func()
      metric_cal = TokenErrMetricCallBack(metric_func, eval_gen, eval_task,
                                          decoder_type)
      callbacks.append(metric_cal)

    logging.info(f"CallBack: Val Metric on {monitor_used}")
    return callbacks

  def get_callbacks(self,
                    eval_ds,
                    eval_task,
                    monitor_used='val_acc',
                    decoder_type='argmax'):
    ''' callbacks for traning, metrics callbacks must be first, then misc callbacks'''
    callbacks = self.get_metric_callbacks(eval_ds, eval_task, monitor_used,
                                          decoder_type)
    misc_cbs = super().get_callbacks(monitor_used)
    callbacks.extend(misc_cbs)
    return callbacks

  def save_model(self):
    ''' save keras model '''
    if self._model_path:
      save_model = self._model_path + str('/final_model.h5')
      self.model.save(save_model)
      logging.info("Model saved: {}".format(save_model))

  def train(self):
    ''' only train '''
    _, train_task = self.input_data(mode=utils.TRAIN)
    self.model_fn(mode=utils.TRAIN)

    callbacks = self.get_misc_callbacks(monitor_used='loss')

    self.active_model.fit_generator(
        train_task,
        steps_per_epoch=len(train_task),
        epochs=self._num_epochs,
        verbose=1,
        callbacks=callbacks,
        max_queue_size=20,
        workers=1,
        use_multiprocessing=False,
        shuffle=True,
        initial_epoch=self._init_epoch)

  def get_metric_func(self):
    ''' build metric function '''
    _input_data = self.model.get_layer('inputs').input
    y_pred = self.model.get_layer('ctc').input[0]
    metric_func = K.function([_input_data], [y_pred])
    return metric_func

  #pylint: disable=too-many-locals
  def eval(self):
    ''' only eval'''
    #get eval dataset
    # data must be init before model build
    logging.info("make Task")
    eval_ds, eval_task = self.input_data(mode=utils.EVAL)
    eval_gen = tf.data.make_one_shot_iterator(eval_ds)

    logging.info("build Model")
    #get eval model
    self.model_fn(mode=utils.EVAL)
    assert self._built

    #load model
    eval_func = self.get_metric_func()

    target_seq_list, predict_seq_list = [], []
    for _ in range(len(eval_task)):
      batch_data = tf.keras.backend.get_session().run(eval_gen.get_next()[0])

      batch_input = batch_data['inputs']
      batch_target = batch_data['targets'].tolist()

      batch_predict = eval_func(batch_input)[0]

      batch_decode = py_ctc.ctc_greedy_decode(batch_predict, 0, unique=True)

      target_seq_list += batch_target
      predict_seq_list += batch_decode

    token_errors = metrics_lib.token_error(
        predict_seq_list=predict_seq_list,
        target_seq_list=target_seq_list,
        eos_id=0)
    logging.info("eval finish!")
    logging.info("Token Error: {}".format(token_errors))

  def train_and_eval(self):
    ''' train and eval '''
    # data must be init before model builg
    #backend_sess = K.get_session()
    train_ds, train_task = self.input_data(mode=utils.TRAIN)
    #train_gen = self.input_generator(tf.data.make_one_shot_iterator(train_ds), train_task, backend_sess, mode=utils.TRAIN)
    eval_ds, eval_task = self.input_data(mode=utils.EVAL)
    #eval_gen = self.input_generator(tf.data.make_one_shot_iterator(eval_ds), eval_task, backend_sess, mode=utils.EVAL)

    self.model_fn(mode=utils.TRAIN)
    assert self._built

    callbacks = self.get_callbacks(
        eval_ds, eval_task, monitor_used=self._monitor_used)

    try:
      # Run training
      self.active_model.fit_generator(
          train_task,
          steps_per_epoch=len(train_task),
          epochs=self._num_epochs,
          verbose=1,
          callbacks=callbacks,
          validation_data=eval_task,
          validation_steps=len(eval_task),
          validation_freq=1,
          class_weight=None,
          max_queue_size=100,
          workers=4,
          use_multiprocessing=False,
          shuffle=True,
          initial_epoch=self._init_epoch)
      #save model
      # not work for subclassed model, using tf.keras.experimental.export_saved_model
      #self.save_model()

    except (Exception, ArithmeticError) as err:  #pylint: disable=broad-except
      template = "An exception of type {0} occurred. Arguments:\n{1!r}"
      message = template.format(type(err).__name__, err.args)
      logging.error(message)
      raise err

    finally:
      # Clear memory
      K.clear_session()
      logging.info("Ending time: {}".format(
          datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

  #pylint: disable=unused-argument,too-many-locals
  def infer(self, yield_single_examples=False):
    ''' only for infer '''
    #load data
    mode = utils.INFER
    # data must be init before model build
    infer_ds, infer_task = self.input_data(mode=mode)
    infer_gen = tf.data.make_one_shot_iterator(infer_ds)

    self.model_fn(mode=mode)
    assert self._built

    #load model
    infer_func = self.get_metric_func()

    for _ in range(len(infer_task)):
      batch_data = tf.keras.backend.get_session().run(infer_gen.get_next()[0])
      batch_input = batch_data['inputs']
      batch_uttid = batch_data['uttids'].tolist()
      batch_predict = infer_func(batch_input)[0]
      batch_decode = py_ctc.ctc_greedy_decode(batch_predict, 0, unique=True)
      for utt_index, uttid in enumerate(batch_uttid):
        logging.info("utt ID: {}".format(uttid))
        logging.info("infer result: {}".format(batch_decode[utt_index]))

  def export_model(self):
    '''export saved_model'''
    mode = utils.INFER
    self.model_fn(mode=mode)
    assert self._built

    input_feat = self.model.get_layer('inputs').input
    input_length = self.model.get_layer('input_length').input

    def ctc_greedy_decode_lambda_func(args):
      y_pred, input_length = args
      input_length = tf.cast(input_length, dtype=tf.int32)
      decode_result, _ = ctc_greedy_decode(
          logits=y_pred,
          sequence_length=input_length,
          merge_repeated=True,
          blank_id=None)
      return decode_result

    model_outputs = self.model.get_layer('outputs').output
    greedy_decode = Lambda(
        ctc_greedy_decode_lambda_func, output_shape=(),
        name='decode')([model_outputs, input_length])

    model_to_export = Model(
        inputs=[input_feat, input_length], outputs=greedy_decode)

    model_export_path = Path(self._model_path).joinpath("export")
    export_saved_model(
        model=model_to_export,
        saved_model_path=str(model_export_path),
        custom_objects=None,
        as_text=False,
        input_signature=None,
        serving_only=False)
