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
''' sklearn metrics '''
import os
import abc
from absl import logging
import numpy as np
from sklearn import metrics
from seqeval.metrics import classification_report as seq_classification_report
from delta.utils.register import registers
from delta.utils.postprocess.postprocess_utils import ids_to_sentences
from rouge import FilesRouge
from sacrebleu import corpus_bleu
from delta.utils.metrics import metric_utils


#pylint: disable=too-few-public-methods
class ABCMetric(metaclass=abc.ABCMeta):
  ''' abstract class of metric '''

  @abc.abstractmethod
  def call(self, y_true=None, y_pred=None, arguments=None):
    ''' compute entrypoint '''
    raise NotImplementedError("Calling an abstract method.")


# pylint: disable=abstract-method
class Metric(ABCMetric):
  ''' wapper of metric '''

  def __init__(self, config):
    self.config = config
    self.pos_label = config['pos_label']

  def __call__(self, *args, **kwargs):
    return self.call(*args, **kwargs)


@registers.metric.register
class AccuracyCal(Metric):
  ''' accuracy metric '''

  # pylint: disable=useless-super-delegation
  def __init__(self, config):
    super().__init__(config)

  def call(self, y_true=None, y_pred=None, arguments=None):
    ''' compute metric '''
    return metrics.accuracy_score(y_true, y_pred, normalize=True)


@registers.metric.register
class F1ScoreCal(Metric):
  ''' F1 score '''

  # pylint: disable=useless-super-delegation
  def __init__(self, config):
    super().__init__(config)

  def call(self, y_true=None, y_pred=None, arguments=None):
    ''' compute metric '''
    average = arguments['average'].lower()
    return metrics.f1_score(y_true, y_pred, average=average)


@registers.metric.register
class PrecisionCal(Metric):
  ''' precision metric '''

  # pylint: disable=useless-super-delegation
  def __init__(self, config):
    super().__init__(config)

  def call(self, y_true=None, y_pred=None, arguments=None):
    ''' compute metric '''
    average = arguments['average'].lower()
    return metrics.precision_score(y_true, y_pred, average=average)


@registers.metric.register
class RecallCal(Metric):
  ''' recall metric '''

  # pylint: disable=useless-super-delegation
  def __init__(self, config):
    super().__init__(config)

  # pylint: disable=too-many-locals
  def call(self, y_true=None, y_pred=None, arguments=None):
    ''' compute metric '''
    average = arguments['average'].lower()
    return metrics.recall_score(y_true, y_pred, average=average)


@registers.metric.register
class ConfusionMatrixCal(Metric):
  ''' confusion matrix '''

  # pylint: disable=useless-super-delegation
  def __init__(self, config):
    super().__init__(config)

  def call(self, y_true=None, y_pred=None, arguments=None):
    ''' compute metric '''
    del arguments
    return metrics.confusion_matrix(y_true, y_pred)


@registers.metric.register
class ClassReportCal(Metric):
  ''' accuracy metric '''

  # pylint: disable=useless-super-delegation
  def __init__(self, config):
    super().__init__(config)

  def call(self, y_true=None, y_pred=None, arguments=None):
    ''' compute metric '''
    return metrics.classification_report(y_true, y_pred)


@registers.metric.register
class TokenErrCal(Metric):
  ''' token error '''

  #pylint: disable=useless-super-delegation
  def __init__(self, config):
    super().__init__(config)

  def call(self, y_true=None, y_pred=None, arguments=None):
    ''' compute metric '''
    eos_id = arguments['eos_id']
    return metric_utils.token_error(
        predict_seq_list=y_pred, target_seq_list=y_true, eos_id=eos_id)


@registers.metric.register
class CrfCal(Metric):
  ''' crf(ner) metric '''

  # pylint: disable=useless-super-delegation
  def __init__(self, config):
    super().__init__(config)

  def call(self, y_true=None, y_pred=None, arguments=None):
    ''' compute metric '''

    label_path_file = arguments["label_vocab_path"]
    return "\n" + seq_classification_report(
        ids_to_sentences(y_true, label_path_file),
        ids_to_sentences(y_pred, label_path_file),
        digits=4)


def run_metrics_for_one_output(metric_config, y_true=None, y_pred=None):
  metrics_list_config = metric_config['cals']
  score = dict()
  for one_metric in metrics_list_config:
    metric_name = one_metric['name']
    calculator = registers.metric[metric_name](metric_config)
    arguments = one_metric['arguments']
    metric_score = calculator(y_true=y_true, y_pred=y_pred, arguments=arguments)
    score[metric_name] = metric_score
  return score


@registers.metric.register
class BleuCal(Metric):
  ''' rouge metric'''

  def __init__(self, config):
    super().__init__(config)
    print(config.keys())
    self.hyp_path = self.config["res_file"]
    self.tgt_paths = self.config['target_file']

  def call(self, y_true=None, y_pred=None, arguments=None):
    with open(self.tgt_paths[0]) as ref, open(self.hyp_path) as hyp:
      bleu = corpus_bleu(hyp, [ref])
      return "\n bleu:" + str(bleu.score)


@registers.metric.register
class RougeCal(Metric):
  ''' rouge metric'''

  def __init__(self, config):
    super().__init__(config)
    print(config.keys())
    self.hyp_path = self.config["res_file"]
    self.ref_path = self.hyp_path + '.gt'
    self.tgt_paths = self.config['target_file']
    self.label_path_file = self.config["text_vocab"]
    self.tgt_paths_after_pre_process = [
        one_path + ".after" for one_path in self.tgt_paths
    ]

  def call(self, y_true=None, y_pred=None, arguments=None):
    ref_sents = []
    for tgt_path in self.tgt_paths_after_pre_process:
      with open(tgt_path, "r", encoding='utf8') as tgt_f:
        ref_sents.extend(tgt_f.readlines())
    ref_sents = [sent.strip() for sent in ref_sents]

    with open(self.ref_path, "w", encoding="utf-8") as in_f:
      for ref_sent in ref_sents:
        in_f.write(ref_sent)
        in_f.write("\n")

    files_rouge = FilesRouge(self.hyp_path, self.ref_path)
    scores = files_rouge.get_scores(avg=True)
    return self.get_scores_output(scores)

  @staticmethod
  def get_scores_output(score_dict):
    res = '\n'
    for rouge_mode in ['rouge-1', 'rouge-2', 'rouge-l']:
      res += '-' * 30 + '\n'
      for metric in ['f', 'p', 'r']:
        res += '{}\tAverage_{}:\t{:.5f}\n'.format(
            rouge_mode.upper(), metric.upper(), score_dict[rouge_mode][metric])
    return res


def get_metrics(config, y_true=None, y_pred=None):
  ''' candies function of metrics
      calc metrics through `y_true` and `y_pred` or
      `confusion` will be deprecated
  '''
  metrics = config['solver']['metrics']
  if isinstance(metrics, list):  # metrics for multi-outputs
    if len(y_true) != len(y_pred) or len(metrics) != len(y_true):
      raise ValueError("Length of y_true, y_pred and metrics must be equal!")
    score = []
    for i, metrics_one_output in enumerate(metrics):
      score.append(
          run_metrics_for_one_output(metrics_one_output, y_true[i], y_pred[i]))
  else:
    score = run_metrics_for_one_output(metrics, y_true, y_pred)
  return score
