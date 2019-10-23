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
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.experimental import export_saved_model

from delta import utils
from delta.utils.decode import py_ctc
from delta.utils import metrics as metrics_lib
from delta.utils.solver.base_solver import Solver
from delta.utils.register import registers
from delta.utils.solver.utils import solver_utils
from delta.utils.solver.utils.callbacks import TokenErrMetricCallBack
from delta.utils.solver.utils.callbacks import ParallelModelCheckpoint
from delta.utils.decode.tf_ctc import ctc_greedy_decode


#pylint: disable=too-many-instance-attributes,too-many-public-methods
@registers.solver.register
class AsrSolver(Solver):
  ''' asr keras solver'''

  def __init__(self, config):
    super().__init__(config)
    self.batch_input_shape = None

    self._solver = config['solver']
    self._num_epochs = self._solver['optimizer']['epochs']

    self._lr = self._solver['optimizer']['learning_rate']['rate']
    self._decay_rate = self._solver['optimizer']['learning_rate']['decay_rate']
    self._val_metric = self._solver['optimizer']['learning_rate'][
        'type'] == 'val_metric'
    if self._val_metric:
      self._min_lr = self._solver['optimizer']['learning_rate']['min_rate']
      self._patience = self._solver['optimizer']['learning_rate']['patience']

    self._clipnorm = self._solver['optimizer']['clip_global_norm']
    self._early_stopping = self._solver['optimizer']['early_stopping']['enable']

    self._monitor_used = self._solver['metrics']['monitor_used']
    self._metrics_used = [] if self._solver['metrics'][
        'metrics_used'] is None else self._solver['metrics']['metrics_used']
    self._model_path = self._solver['saver']['model_path']
    self._model_load_type = self._solver['loader']['model_load_type']
    self._init_epoch = self._solver['loader']['init_epoch']
    self._specified_model_file = self._solver['loader']['file_name']

    self._checkpoint_file_pattern = 'model.{epoch:02d}-{monitor_used:.2f}.ckpt'

    logging.info('num_epochs : {}'.format(self._num_epochs))
    logging.info('lr : {}'.format(self._lr))
    logging.info('saver path : {}'.format(self._model_path))

    devices, self._ngpu = utils.gpu_device_names()
    logging.info(f"ngpu: {self._ngpu}, device list: {devices}")

    #model
    self._model = None
    self._parallel_model = None
    self._built = False

  @property
  def ngpu(self):
    ''' number of gpus '''
    return self._ngpu

  @property
  def raw_model(self):
    ''' Delta RawModel '''
    assert self._model is not None
    return self._model

  @property
  def model(self):
    ''' keras Model before doing `multi_gpu_model` '''
    return self.raw_model.model

  @property
  def parallel_model(self):
    ''' `multi_gpu_model` of keras Model '''
    assert self._parallel_model is not None
    return self._parallel_model

  @property
  def active_model(self):
    ''' real keras model for run'''
    return self.parallel_model if self.ngpu > 1 else self.model

  def process_config(self, config):
    ''' preprocess of config'''
    return config

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

  def input_generator(self, input_iterator, input_task, cur_sess, mode):
    ''' dataset_based generator used in keras.model.fit_generator()
        in future, it will be replaced by tf.keras.utils.Sequence'''
    next_batch = input_iterator.get_next()
    generate_time = len(
        input_task) * self._num_epochs if mode == utils.TRAIN else len(
            input_task)
    for _ in range(generate_time):
      next_batch_data = cur_sess.run(next_batch)
      yield next_batch_data

  def get_run_opts_metas(self):
    ''' RunOptions and RunMetadata '''
    opts_conf = self.config['solver']['run_options']
    run_opts = tf.RunOptions(
        trace_level=opts_conf['trace_level'],
        inter_op_thread_pool=opts_conf['inter_op_thread_pool'],
        report_tensor_allocations_upon_oom=opts_conf[
            'report_tensor_allocations_upon_oom'])
    run_metas = tf.RunMetadata()
    run_metas = None
    run_opts = None
    return run_opts, run_metas

  def get_optimizer(self, multitask):
    ''' keras optimizer '''
    optconf = self.config['solver']['optimizer']
    method = optconf['name']

    learning_rate = optconf['learning_rate']['rate']
    if method == 'adam':
      opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif method == 'adadelta':
      opt = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
    else:
      raise ValueError(f"Not support optimmizer: {method}")
    return opt

  #pylint: disable=arguments-differ
  def model_fn(self, mode):
    ''' build model like tf.estimator.Estimator'''
    with tf.device('/cpu:0'):
      self._model = super().model_fn()

    if not self.model.built:
      assert self.batch_input_shape
      # data must be (features, labels), only using features as input
      self.model.build(input_shape=self.batch_input_shape[0])

    assert self._init_epoch in range(0, self._num_epochs)
    model_load_type, model_file_name = solver_utils.get_model_file(
        dir_name=self._model_path,
        file_name_pattern=self._checkpoint_file_pattern,
        mode=mode,
        model_load_type=self._model_load_type,
        specified_model_file_name=self._specified_model_file)

    logging.info("{}-{}: load model from {}".format(mode, model_load_type,
                                                    model_file_name))
    if model_file_name is not None:
      if self.model.built:
        self.model.load_weights(str(model_file_name), by_name=False)
      else:
        self._model = tf.keras.models.load_model(str(model_file_name))

    # parallel and compile model
    self.build(multi_gpu=(mode == utils.TRAIN))

  def build(self, multi_gpu=False):
    ''' main entrypoint to build model '''
    assert self.model

    loss = self.get_loss()
    multitask = self.config['solver']['optimizer']['multitask']
    optimizer = self.get_optimizer(multitask)

    run_opts, run_metas = self.get_run_opts_metas()

    # compile model
    if self.ngpu > 1 and multi_gpu:
      self._parallel_model = multi_gpu_model(
          self.model, gpus=self.ngpu, cpu_relocation=False, cpu_merge=False)
      self.parallel_model.compile(
          loss=loss,
          optimizer=optimizer,
          metrics=self._metrics_used,
          options=run_opts,
          run_metadata=run_metas)
    else:
      self.model.compile(
          loss=loss,
          optimizer=optimizer,
          metrics=self._metrics_used,
          options=run_opts,
          run_metadata=run_metas)

    # Print model summary
    if self.model.built and self.model._is_graph_network:
      self.model.summary()
    self._built = True

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

  def get_misc_callbacks(self, monitor_used=None):
    '''misc_specific callbacks'''
    callbacks = []
    #tensorboard
    tb_cb = TensorBoard(log_dir=self._model_path)
    callbacks.append(tb_cb)
    logging.info(f"CallBack: Tensorboard")

    # metric history
    metric_log = 'metrics.csv'
    csv_logger = CSVLogger(
        filename=Path(self._model_path).joinpath(metric_log), separator='\t')
    callbacks.append(csv_logger)
    logging.info(f"CallBack: Metric log to {metric_log}")

    #save model
    save_best = Path(self._model_path).joinpath('best_model.ckpt')
    save_best_cb = ParallelModelCheckpoint(
        model=self.model,
        filepath=str(save_best),
        monitor=monitor_used,
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        period=1)
    callbacks.append(save_best_cb)
    logging.info(f"CallBack: Save Best Model")

    # save checkpoint
    save_file_pattern = self._checkpoint_file_pattern.replace(
        'monitor_used', monitor_used)
    save_ckpt = Path(self._model_path).joinpath(save_file_pattern)
    save_ckpt_cb = ParallelModelCheckpoint(
        model=self.model,
        filepath=str(save_ckpt),
        monitor=monitor_used,
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        period=1)
    callbacks.append(save_ckpt_cb)
    logging.info(f"CallBack: Save Model Checkpoint.")

    # nan check
    callbacks.append(tf.keras.callbacks.TerminateOnNaN())

    # Stops the model early if the metrics isn't improving
    if self._early_stopping:
      logging.info(f"CallBack: Early Stop on {monitor_used}")
      es_cb = EarlyStopping(
          monitor=monitor_used, min_delta=0, patience=5, verbose=0, mode='auto')
      callbacks.append(es_cb)

    # shcedule  learning rate
    if self._val_metric:
      logging.info(f"CallBack: Learning Rate Shcedule on {monitor_used}")
      lr_shcedule = ReduceLROnPlateau(
          monitor=monitor_used,
          factor=self._decay_rate,
          patience=self._patience,
          verbose=1,
          mode='auto',
          min_delta=0.0001,
          cooldown=0,
          min_lr=self._min_lr)
      callbacks.append(lr_shcedule)
    return callbacks

  def get_callbacks(self,
                    eval_ds,
                    eval_task,
                    monitor_used='val_acc',
                    decoder_type='argmax'):
    ''' callbacks for traning'''

    #metric callbacks
    callbacks = self.get_metric_callbacks(eval_ds, eval_task, monitor_used,
                                          decoder_type)
    #misc callbacks
    misc_callbacks = self.get_misc_callbacks(monitor_used)
    callbacks.extend(misc_callbacks)

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
      batch_data = K.get_session().run(eval_gen.get_next()[0])

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
      batch_data = K.get_session().run(infer_gen.get_next()[0])
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
