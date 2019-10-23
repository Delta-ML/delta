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
''' Estimator base class for classfication '''
import os
import functools
from absl import logging
import delta.compat as tf
from tensorflow.python import debug as tf_debug  #pylint: disable=no-name-in-module
from tensorflow.python.estimator.canned import metric_keys
# See: tensorboard/tensorboard/plugins/pr_curve/README.md
# Note: tf.contrib.metrics.streaming_curve_points will only produce a series of points
#  which won't be displayed by Tensorboard.
from tensorboard.plugins.pr_curve import summary as pr_summary

from delta import utils
from delta.utils.hparam import HParams
from delta.utils import metrics as metrics_lib
from delta.utils import summary as summary_lib
from delta.utils.register import registers
from delta.utils.solver.base_solver import ABCEstimatorSolver


#pylint: disable=abstract-method
class EstimatorSolver(ABCEstimatorSolver):
  ''' base class for classfication '''

  def __init__(self, config):
    super().__init__(config)

    saver_conf = config['solver']['saver']
    self.eval_path = os.path.join(saver_conf['model_path'], 'eval')
    if not os.path.exists(self.eval_path):
      os.makedirs(self.eval_path)
    self.eval_metrics_path = os.path.join(self.eval_path, 'metrics.txt')

  def input_fn(self, mode):
    ''' return input_fn '''
    super().input_fn(mode)
    batch_size = self.config['solver']['optimizer']['batch_size']
    num_epoch = self.config['solver']['optimizer']['epochs']
    return self.task.input_fn(mode, batch_size, num_epoch)

  def get_scaffold(self, mode, global_step=None):
    if mode != utils.TRAIN:
      # for model average
      saver = self.get_saver(global_step)
      scaffold = tf.train.Scaffold(saver=saver)
    else:
      scaffold = None  # default
    return scaffold

  def l2_loss(self, tvars=None):
    _l2_loss = 0.0
    weight_decay = self.config['solver']['optimizer'].get('weight_decay', None)
    if weight_decay:
      logging.info(f"add L2 Loss with decay: {weight_decay}")
      with tf.name_scope('l2_loss'):
        tvars = tvars if tvars else tf.trainable_variables()
        tvars = [v for v in tvars if 'bias' not in v.name]
        _l2_loss = weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in tvars])
        summary_lib.scalar('l2_loss', _l2_loss)
    return _l2_loss

  def model_fn(self):
    ''' return model_fn '''
    model_class = super().model_fn()
    model_name = self.config['model']['name']
    model_type = self.config['model']['type']
    logging.info("Model: {}, {}".format(model_type, model_name))

    #pylint: disable=too-many-locals
    def _model_fn(features, labels, mode, params):
      del params
      is_train = mode == utils.TRAIN

      # Supports both dict output and legacy single logits output.
      model_outputs = model_class(features, training=is_train)
      if isinstance(model_outputs, dict):
        logits = model_outputs['logits']
        extra_outputs = dict(model_outputs)
        extra_outputs.pop('logits')
      else:
        logits = model_outputs
        extra_outputs = None

      alignment = model_class.alphas if hasattr(model_class, 'alphas') else None

      if mode == utils.INFER:
        softmax = tf.nn.softmax(logits, name='softmax_output')
        predictions = self.get_infer_predictions(
            features, softmax, alpha=alignment, extra_outputs=extra_outputs)
        return tf.estimator.EstimatorSpec( #pylint: disable=no-member
            mode=mode,
            predictions=predictions,
            scaffold=self.get_scaffold(mode),
            export_outputs={
                'predictions': tf.estimator.export.PredictOutput(softmax) #pylint: disable=no-member
            })

      if 'soft_labels' in features.keys():
        soft_labels = features['soft_labels']
      else:
        soft_labels = None

      loss = self.get_loss_fn()(
          labels=labels,
          logits=logits,
          soft_labels=soft_labels,
          name='x_loss',
      )

      if mode == utils.TRAIN:  #pylint: disable=no-else-return
        multitask = self.config['solver']['optimizer']['multitask']
        if self.config['solver']['adversarial']['enable']:
          x = features['inputs']  #pylint: disable=invalid-name
          grad, = tf.gradients(loss, x)
          x_adv = x + self.config['solver']['adversarial'][
              'adv_epslion'] * tf.sign(grad)
          x_adv = tf.stop_gradient(x_adv)
          features_adv = {'inputs': x_adv, 'texts': features['text']}
          logits_adv = model_class(features_adv)

          loss_adv = self.get_loss_fn()(
              labels=labels,
              logits=logits_adv,
              soft_labels=soft_labels,
              name='x_adv_loss',
          )
          adv_alpha = self.config['solver']['adversarial']['adv_alpha']
          loss_all = (1 - adv_alpha) * loss + adv_alpha * loss_adv
          multitask = True
        else:
          loss_all = loss

        # L2 loss
        loss_all += self.l2_loss()

        train_op = self.get_train_op(loss_all, multitask=multitask)
        train_hooks = self.get_train_hooks(labels, logits, alpha=alignment)

        utils.log_vars('Global Vars', tf.global_variables())
        return tf.estimator.EstimatorSpec(  #pylint: disable=no-member
            mode=mode,
            loss=loss_all,
            train_op=train_op,
            training_chief_hooks=train_hooks,
            training_hooks=None,
            scaffold=None,
        )
      else:  # eval
        loss_all = loss
        eval_hooks, eval_metrics_ops = self.get_eval_hooks(labels, logits)
        return tf.estimator.EstimatorSpec(  #pylint: disable=no-member
            mode=mode,
            loss=loss_all,
            eval_metric_ops=eval_metrics_ops,
            evaluation_hooks=eval_hooks,
            scaffold=self.get_scaffold(mode),
        )

    return _model_fn

  def create_estimator(self):
    # Set model params
    model_params = HParams()

    # create model func
    model_fn = self.model_fn()

    # multi-gpus
    devices, num_gpu = utils.gpu_device_names()
    distribution = utils.get_distribution_strategy(num_gpu)
    logging.info('Device: {}/{}'.format(num_gpu, devices))

    # run config
    tfconf = self.config['solver']['run_config']
    saverconf = self.config['solver']['saver']
    session_config = tf.ConfigProto(
        allow_soft_placement=tfconf['allow_soft_placement'],
        log_device_placement=tfconf['log_device_placement'],
        intra_op_parallelism_threads=tfconf['intra_op_parallelism_threads'],
        inter_op_parallelism_threads=tfconf['inter_op_parallelism_threads'],
        gpu_options=tf.GPUOptions(allow_growth=tfconf['allow_growth']))

    run_config = tf.estimator.RunConfig(  #pylint: disable=no-member
        tf_random_seed=tfconf['tf_random_seed'],
        session_config=session_config,
        save_summary_steps=saverconf['save_summary_steps'],
        keep_checkpoint_max=saverconf['max_to_keep'],
        log_step_count_steps=tfconf['log_step_count_steps'],
        train_distribute=distribution,
        device_fn=None,
        protocol=None,
        eval_distribute=None,
        experimental_distribute=None,
    )

    # Instantiate Estimator
    nn = tf.estimator.Estimator(  #pylint: disable=no-member,invalid-name
        model_fn=model_fn,
        model_dir=saverconf['model_path'],
        config=run_config,
        params=model_params,
        warm_start_from=None,
    )
    return nn

  def get_train_hooks(self, labels, logits, alpha=None):
    nclass = self.config['data']['task']['classes']['num']
    metric_tensor = {
        "batch_accuracy": metrics_lib.accuracy(logits, labels),
        'global_step': tf.train.get_or_create_global_step(),
    }
    if nclass > 100:
      logging.info('Too many classes, disable confusion matrix in train: %d' %
                   (nclass))
    else:
      metric_tensor['batch_confusion'] = \
          metrics_lib.confusion_matrix(logits, labels, nclass)
    summary_lib.scalar('batch_accuracy', metric_tensor['batch_accuracy'])
    if alpha:
      metric_tensor.update({"alignment": alpha})

    # plot PR curve
    true_label_bool = tf.cast(labels, tf.bool)
    softmax = tf.nn.softmax(logits)
    pr_summary.op(
        name='pr_curve_train_batch',
        labels=true_label_bool,
        predictions=softmax[:, -1],
        num_thresholds=16,
        weights=None)

    train_hooks = [
        tf.train.StepCounterHook(
            every_n_steps=100,
            every_n_secs=None,
            output_dir=None,
            summary_writer=None),
        tf.train.FinalOpsHook(
            final_ops=[tf.train.get_or_create_global_step()],
            final_ops_feed_dict=None),
        tf.train.LoggingTensorHook(
            tensors=metric_tensor,
            every_n_iter=100,
            every_n_secs=None,
            at_end=False,
            formatter=None),
    ]
    return train_hooks

  def get_eval_hooks(self, labels, logits):
    ''' lables: [batch]
        logits: [batch, num_classes]
    '''
    nclass = self.config['data']['task']['classes']['num']

    eval_hooks = []
    metric_tensor = {}
    with tf.variable_scope('metrics'):
      true_label = labels
      true_label_bool = tf.cast(labels, tf.bool)
      softmax = tf.nn.softmax(logits)
      pred_label = tf.argmax(softmax, -1)
      eval_metrics_ops = {
          'accuracy':
              tf.metrics.accuracy(
                  labels=true_label, predictions=pred_label, weights=None),
      }
      if nclass == 2:
        eval_metrics_ops.update({
            'auc':
                tf.metrics.auc(
                    labels=true_label,
                    predictions=softmax[:, -1],
                    num_thresholds=20,
                    curve='ROC',
                    summation_method='trapezoidal'),
            'precision':
                tf.metrics.precision(
                    labels=true_label, predictions=pred_label, weights=None),
            'recall':
                tf.metrics.recall(
                    labels=true_label, predictions=pred_label, weights=None),
            'tp':
                tf.metrics.true_positives(
                    labels=true_label, predictions=pred_label, weights=None),
            'fn':
                tf.metrics.false_negatives(
                    labels=true_label, predictions=pred_label, weights=None),
            'fp':
                tf.metrics.false_positives(
                    labels=true_label, predictions=pred_label, weights=None),
            'tn':
                tf.metrics.true_negatives(
                    labels=true_label, predictions=pred_label, weights=None),
            'pr_curve_eval':
                pr_summary.streaming_op(
                    name='pr_curve_eval',
                    labels=true_label_bool,
                    predictions=softmax[:, -1],
                    num_thresholds=50,
                    weights=None)
        })

    metric_tensor.update({key: val[0] for key, val in eval_metrics_ops.items()})
    metric_hook = tf.train.LoggingTensorHook(
        tensors=metric_tensor,
        every_n_iter=100,
        every_n_secs=None,
        at_end=False,
        formatter=None)
    eval_hooks.append(metric_hook)
    return eval_hooks, eval_metrics_ops

  #pylint: disable=arguments-differ
  def get_infer_predictions(self,
                            features,
                            softmax,
                            alpha=None,
                            extra_outputs=None):
    '''
    Get prediction results for postprocessing at inference time.
    Args:
      extra_outputs: optional extra outputs from model() other than logits.
    '''
    predictions = {"softmax": softmax}
    if alpha:
      predictions.update({
          "alignment": alpha,
      })
    predictions.update(features)
    predictions.pop('audio', None)
    if extra_outputs is not None:
      predictions.update(extra_outputs)

    logging.info('predictions: {}'.format(predictions))
    return predictions

  def train_one_epoch(self, nn, steps=None):  #pylint: disable=invalid-name
    ''' train for one epoch '''
    mode = utils.TRAIN
    tfconf = self.config['solver']['run_config']
    nn.train(
        input_fn=self.input_fn(mode),
        steps=steps,
        hooks=[tf_debug.LocalCLIDebugHook()] if tfconf['debug'] else None)

  def train(self):
    ''' only train '''
    nn = self.create_estimator()  #pylint: disable=invalid-name

    num_epochs = self.config['solver']['optimizer']['epochs']
    for epoch in range(num_epochs):
      logging.info("epoch: {}".format(epoch + 1))
      try:
        self.train_one_epoch(nn)
      except tf.errors.OutOfRangeError as err:
        logging.info(epoch, err)
        raise err

  def log_eval_metrics(self, metrics, save=False):
    ''' eval metrics '''
    logging.info('Eval Result:')
    if save:
      fobj = open(self.eval_metrics_path, 'w')
    for key, value in metrics.items():
      logging.info("{}: {}".format(key, value))
      if save:
        fobj.write("{}: {}\n".format(key, value))
    if 'tp' in metrics and 'fp' in metrics and 'fn' in metrics and 'tn' in metrics:
      f1_score = metrics_lib.f1_score(metrics['tp'], metrics['fp'],
                                      metrics['fn'], metrics['tn'])
      logging.info("F1: {}".format(f1_score))
      if save:
        fobj.write("F1: {}\n".format(f1_score))

  #pylint: disable=arguments-differ
  def eval(self, steps=None):
    ''' only eval '''
    mode = utils.EVAL
    nn = self.create_estimator()  #pylint: disable=invalid-name
    ev = nn.evaluate(  #pylint: disable=invalid-name
        input_fn=self.input_fn(mode),
        steps=steps,
        hooks=None,
        checkpoint_path=None,
        name=None)
    self.log_eval_metrics(ev, save=True)

  #pylint: disable=invalid-name
  def train_and_eval_one_epoch(self, nn, train_spec, eval_spec):
    ''' train and eval for one epoch '''
    eval_result, export_result = tf.estimator.train_and_evaluate(  #pylint: disable=no-member
        nn, train_spec, eval_spec)
    logging.info("Export result:{}".format(export_result))
    self.log_eval_metrics(eval_result)

  def train_and_eval(self):
    ''' train and eval '''
    nn = self.create_estimator()  #pylint: disable=invalid-name
    #logging.info("Vars: {}".format(nn.get_variable_names()))

    #pylint: disable=no-member
    train_spec = tf.estimator.TrainSpec(
        input_fn=self.input_fn(utils.TRAIN), max_steps=None, hooks=None)

    #pylint: disable=no-member
    nclass = self.config['data']['task']['classes']['num']
    # https://github.com/tensorflow/estimator/blob/master/tensorflow_estimator/python/estimator/canned/metric_keys.py
    if nclass == 2:
      default_key = metric_keys.MetricKeys.AUC
    else:
      default_key = metric_keys.MetricKeys.ACCURACY
    compare_fn = functools.partial(
        utils.metric_smaller, default_key=default_key)
    logging.info(f"Using {default_key} metric for best exporter")

    eval_spec = tf.estimator.EvalSpec(
        input_fn=self.input_fn(utils.EVAL),
        steps=None,
        name='dev',
        hooks=None,
        exporters=[
            tf.estimator.FinalExporter(
                name='final_export',
                serving_input_receiver_fn=self.create_serving_input_receiver_fn(
                ),
                assets_extra=None,
                as_text=False),
            tf.estimator.BestExporter(
                name='best_exporter',
                serving_input_receiver_fn=self.create_serving_input_receiver_fn(
                ),
                event_file_pattern='eval/*.tfevents.*',
                compare_fn=compare_fn,
                assets_extra=None,
                as_text=False,
                exports_to_keep=1,
            ),
        ],
        start_delay_secs=60,
        throttle_secs=600)

    num_epochs = self.config['solver']['optimizer']['epochs']
    for epoch in range(num_epochs):
      logging.info("epoch: {}".format(epoch + 1))
      try:
        self.train_and_eval_one_epoch(nn, train_spec, eval_spec)
      except tf.errors.OutOfRangeError as err:
        logging.info(epoch, err)
        raise err

  #pylint: disable=arguments-differ
  def infer(self, yield_single_examples=True):
    nn = self.create_estimator()  #pylint: disable=invalid-name
    predictions = nn.predict(
        input_fn=self.input_fn(utils.INFER),
        predict_keys=None,
        hooks=None,
        checkpoint_path=None,
        yield_single_examples=yield_single_examples,
    )
    return self.postproc_fn()(predictions, log_verbose=False)

  def export_model(self):
    saver_conf = self.config['solver']['saver']
    nn = self.create_estimator()  #pylint: disable=invalid-name
    nn.export_savedmodel(
        export_dir_base=os.path.join(saver_conf['model_path'], 'export'),
        serving_input_receiver_fn=self.create_serving_input_receiver_fn(),
        assets_extra=None,
        as_text=False,
        checkpoint_path=None,
        strip_default_attrs=False,
    )

  def postproc_fn(self):
    postproc_name = self.config['solver']['postproc']["name"]
    postproc = registers.postprocess[postproc_name](self.config)

    return postproc
