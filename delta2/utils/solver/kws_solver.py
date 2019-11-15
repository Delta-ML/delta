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
''' kws solver based on EstimatorSolver'''
import delta.compat as tf

from delta.utils import metrics
from delta.utils.register import registers
from delta.utils.solver.estimator_solver import EstimatorSolver


@registers.solver.register
class KwsSolver(EstimatorSolver):
  ''' kws solver'''

  #pylint: disable=useless-super-delegation
  def __init__(self, config):
    super().__init__(config)

  def process_config(self, config):
    data_conf = config['data']
    class_vocab = data_conf['task']['classes']['vocab']
    assert len(class_vocab) == data_conf['task']['classes']['num']

    # add revere_vocab, positive_id
    reverse_vocab = {val: key for key, val in class_vocab.items()}
    data_conf['task']['classes']['reverse_vocab'] = reverse_vocab

    # add feature shape, without batch_size
    feat_dim = data_conf['task']['audio']['feat_dim']
    if data_conf['task']['audio']['splice_frame']:
      feat_dim = feat_dim * (data_conf['task']['audio']['left_context'] \
              + 1 + data_conf['task']['audio']['right_context'])
    final_feat_dim = feat_dim * (data_conf['task']['audio']['delta_order'] + 1)
    window_len = data_conf['task']['audio']['window_len']
    feature_shape = [window_len, final_feat_dim]
    data_conf['task']['audio']['feature_shape'] = feature_shape
    return config

  def get_train_hooks(self, labels, logits, alpha=None):
    nclass = self.config['data']['task']['classes']['num']
    metric_tensor = {
        'global_step': tf.train.get_or_create_global_step(),
        "batch_accuracy": metrics.accuracy(logits, labels),
        "batch_confusion": metrics.confusion_matrix(logits, labels, nclass),
    }
    tf.summary.scalar('batch_accuracy', metric_tensor['batch_accuracy'])

    train_hooks = [
        tf.train.StepCounterHook(
            every_n_steps=500,
            every_n_secs=None,
            output_dir=None,
            summary_writer=None),
        tf.train.FinalOpsHook(
            final_ops=[tf.train.get_or_create_global_step()],
            final_ops_feed_dict=None),
        tf.train.LoggingTensorHook(
            tensors=metric_tensor,
            every_n_iter=1000,
            every_n_secs=None,
            at_end=False,
            formatter=None),
    ]
    return train_hooks

  def get_eval_hooks(self, labels, logits):
    ''' lables: [batch]
            logits: [batch, num_classes]
        '''
    eval_hooks = []
    metric_tensor = {}
    with tf.variable_scope('metrics'):
      true_label = labels
      softmax = tf.nn.softmax(logits)
      pred_label = tf.argmax(softmax, -1)
      eval_metrics_ops = {
          'accuracy':
              tf.metrics.accuracy(
                  labels=true_label, predictions=pred_label, weights=None),
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
      }

    metric_tensor.update({key: val[0] for key, val in eval_metrics_ops.items()})
    metric_hook = tf.train.LoggingTensorHook(
        tensors=metric_tensor,
        every_n_iter=10000,
        every_n_secs=None,
        at_end=False,
        formatter=None)
    eval_hooks.append(metric_hook)
    return eval_hooks, eval_metrics_ops

  def create_serving_input_receiver_fn(self):
    # shape must be with batch_size
    taskconf = self.config['data']['task']
    shape = taskconf['audio']['feature_shape']
    shape.insert(0, None)

    return tf.estimator.export.build_raw_serving_input_receiver_fn(  #pylint:disable=no-member
        features={
            'inputs':
                tf.placeholder(name="inputs", shape=shape, dtype=tf.float32),
        },
        default_batch_size=None,
    )
