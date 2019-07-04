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
import tensorflow as tf
#pylint: disable=import-error
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard

from delta import utils
from delta.utils.solver.base_solver import Solver
from delta.utils.register import registers


#pylint: disable=too-many-instance-attributes
@registers.solver.register
class AsrSolver(Solver):
  ''' asr keras solver'''

  def __init__(self, config):
    super().__init__(config)
    self._solver = config['solver']
    self._num_epochs = self._solver['optimizer']['epochs']

    self._lr = self._solver['optimizer']['learning_rate']['rate']
    self._decay = self._solver['optimizer']['learning_rate']['decay_rate']
    self._clipnorm = self._solver['optimizer']['clip_global_norm']
    self._optimizer = self._solver['optimizer']['name']
    self._early_stopping = self._solver['optimizer']['early_stopping']['enable']

    self._model_path = self._solver['saver']['model_path']

    logging.info('num_epochs : {}'.format(self._num_epochs))
    logging.info('lr : {}'.format(self._lr))
    logging.info('saver path : {}'.format(self._model_path))

    _, self._ngpu = utils.gpu_device_names()

    #model
    self._model = None
    self._parallel_model = None
    self._built = False

  def process_config(self, config):
    ''' preprocess of config'''
    return config

  @property
  def ngpu(self):
    ''' number of gpus '''
    return self._ngpu

  def input_fn(self, mode):
    ''' input function for tf.data.Dataset'''
    super().input_fn(mode)
    batch_size = self.config['solver']['optimizer']['batch_size']
    num_epoch = self.config['solver']['optimizer']['epochs']
    return self.task.input_fn(mode, batch_size, num_epoch), self.task

  #pylint: disable=arguments-differ
  def model_fn(self, mode):
    ''' build model like tf.estimator.Estimator'''
    self._model = super().model_fn()
    self.build(self.model, multi_gpu=mode == utils.TRAIN)

  @property
  def raw_model(self):
    ''' Delta Model '''
    assert self._model is not None
    return self._model

  @property
  def model(self):
    ''' keras Model '''
    return self.raw_model.model

  @property
  def parallel_model(self):
    ''' multi_gpu_model of keras Model '''
    assert self._parallel_model is not None
    return self._parallel_model

  @property
  def active_model(self):
    ''' real keras model for run'''
    return self.parallel_model if self.ngpu > 1 else self.model

  #pylint: disable=no-self-use
  def get_ctc_loss(self):
    ''' dummy ctc loss, since ctc is implemented as a kearas layer '''
    loss = {'ctc': lambda y_true, y_pred: y_pred}
    return loss

  def input(self, mode):
    ''' input data '''
    input_fn, task = self.input_fn(mode)
    ds_ = input_fn()
    iterator = ds_.make_one_shot_iterator()
    return iterator, task

  def get_run_opts_metas(self):
    ''' RunOptions and RunMetadata '''
    opts_conf = self.config['solver']['run_options']
    run_opts = tf.RunOptions(
        trace_level=opts_conf['trace_level'],
        inter_op_thread_pool=opts_conf['inter_op_thread_pool'],
        report_tensor_allocations_upon_oom=opts_conf[
            'report_tensor_allocations_upon_oom'])
    run_metas = tf.RunMetadata()
    return run_opts, run_metas

  def build(self, model, multi_gpu=True):
    ''' main entrypoint to build model '''
    loss = self.get_ctc_loss()
    multitask = self.config['solver']['optimizer']['multitask']
    optimizer = self.get_optimizer(multitask)
    optimizer = Adam()

    run_opts, run_metas = self.get_run_opts_metas()

    # compile model
    if self.ngpu > 1 and multi_gpu:
      self._parallel_model = multi_gpu_model(model, gpus=self.ngpu)
      self.parallel_model.compile(
          loss=loss,
          optimizer=optimizer,
          options=run_opts,
          run_metadata=run_metas)
    else:
      model.compile(
          loss=loss,
          optimizer=optimizer,
          options=run_opts,
          run_metadata=run_metas)

    # Print model summary
    model.summary()
    self._built = True

  def get_callbacks(self, mode):
    ''' callbacks for traning'''
    callbacks = []
    #tensorboard
    tb_cb = TensorBoard(log_dir=self._model_path)
    callbacks.append(tb_cb)

    if mode == utils.TRAIN:
      #save best
      save_best = Path(self._model_path).joinpath('best_model.h5')
      save_best_cb = ModelCheckpoint(
          str(save_best),
          monitor='val_loss',
          verbose=1,
          save_best_only=True,
          period=1)
      callbacks.append(save_best_cb)
      save_ckpt = Path(
          self._model_path).joinpath('model.{epoch:02d}-{val_loss:.2f}.h5')
      save_ckpt_cb = ModelCheckpoint(
          str(save_ckpt),
          monitor='val_loss',
          verbose=1,
          save_best_only=False,
          period=1)
      callbacks.append(save_ckpt_cb)
      # nan check
      callbacks.append(tf.keras.callbacks.TerminateOnNaN())
      # Stops the model early if the val_loss isn't improving
      if self._early_stopping:
        es_cb = EarlyStopping(
            monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
        callbacks.append(es_cb)
    return callbacks

  def save_model(self):
    ''' save keras model '''
    if self._model_path:
      save_model = self._model_path + str('/final_model.h5')
      self.model.save(save_model)
      logging.info("Model saved: {}".format(save_model))

  def train(self):
    ''' only train '''
    mode = utils.TRAIN
    self.model_fn(mode)
    assert self._built
    train_gen, train_task = self.input(mode=utils.TRAIN)
    callbacks = self.get_callbacks(mode=utils.TRAIN)
    self.active_model.fit(
        x=train_gen,
        steps_per_epoch=train_task.steps_per_epoch,
        epochs=self._num_epochs,
        verbose=1,
        callbacks=callbacks)

  def get_eval_model(self):
    ''' build eval model '''

  def eval(self):
    ''' only eval'''
    mode = utils.EVAL
    self.model_fn(mode)
    assert self._built
    eval_gen, eval_task = self.input(mode=mode)
    self.active_model.evaluate(
        x=eval_gen, steps=eval_task.steps_per_epoch, verbose=1)

  def train_and_eval(self):
    ''' train and eval '''
    self.model_fn(mode=utils.TRAIN)
    assert self._built
    train_gen, train_task = self.input(mode=utils.TRAIN)
    eval_gen, eval_task = self.input(mode=utils.EVAL)
    callbacks = self.get_callbacks(mode=utils.TRAIN)

    try:
      # Run training
      self.active_model.fit(
          x=train_gen,
          steps_per_epoch=train_task.steps_per_epoch,
          epochs=self._num_epochs,
          verbose=1,
          callbacks=callbacks,
          validation_data=eval_gen,
          validation_steps=eval_task.steps_per_epoch)
      #save model
      self.save_model()

    except (Exception, ArithmeticError) as err:  #pylint: disable=broad-except
      template = "An exception of type {0} occurred. Arguments:\n{1!r}"
      message = template.format(type(err).__name__, err.args)
      logging.error(message)

    finally:
      # Clear memory
      K.clear_session()
      logging.info("Ending time: {}".format(
          datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

  def infer(self):
    ''' only for infer '''

  def export_model(self):
    ''' export saved_model '''
