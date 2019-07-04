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
import abc
from sklearn import metrics
from seqeval.metrics import classification_report as seq_classification_report
from delta.utils.register import registers
from delta.utils.postprocess.postprocess_utils import ids_to_sentences

#pylint: disable=too-few-public-methods
class ABCMetric(metaclass=abc.ABCMeta):
  ''' abstract class of metric '''

  @abc.abstractmethod
  def call(self, y_true=None, y_pred=None, arguments=None):
    ''' compute entrypoint '''
    raise NotImplementedError("Calling an abstract method.")


#pylint: disable=abstract-method
class Metric(ABCMetric):
  ''' wapper of metric '''

  def __init__(self, config):
    self.config = config
    self.pos_label = config['solver']['metrics']['pos_label']

  def __call__(self, *args, **kwargs):
    return self.call(*args, **kwargs)


@registers.metric.register
class AccuracyCal(Metric):
  ''' accuracy metric '''

  #pylint: disable=useless-super-delegation
  def __init__(self, config):
    super().__init__(config)

  def call(self, y_true=None, y_pred=None, arguments=None):
    ''' compute metric '''
    return metrics.accuracy_score(y_true, y_pred, normalize=True)


@registers.metric.register
class F1ScoreCal(Metric):
  ''' F1 score '''

  #pylint: disable=useless-super-delegation
  def __init__(self, config):
    super().__init__(config)

  def call(self, y_true=None, y_pred=None, arguments=None):
    ''' compute metric '''
    average = arguments['average'].lower()
    return metrics.f1_score(y_true, y_pred, average=average)


@registers.metric.register
class PrecisionCal(Metric):
  ''' precision metric '''

  #pylint: disable=useless-super-delegation
  def __init__(self, config):
    super().__init__(config)

  def call(self, y_true=None, y_pred=None, arguments=None):
    ''' compute metric '''
    average = arguments['average'].lower()
    return metrics.precision_score(y_true, y_pred, average=average)


@registers.metric.register
class RecallCal(Metric):
  ''' recall metric '''

  #pylint: disable=useless-super-delegation
  def __init__(self, config):
    super().__init__(config)

  #pylint: disable=too-many-locals
  def call(self, y_true=None, y_pred=None, arguments=None):
    ''' compute metric '''
    average = arguments['average'].lower()
    return metrics.recall_score(y_true, y_pred, average=average)


@registers.metric.register
class ConfusionMatrixCal(Metric):
  ''' confusion matrix '''

  #pylint: disable=useless-super-delegation
  def __init__(self, config):
    super().__init__(config)

  def call(self, y_true=None, y_pred=None, arguments=None):
    ''' compute metric '''
    del arguments
    return metrics.confusion_matrix(y_true, y_pred)


@registers.metric.register
class ClassReportCal(Metric):
  ''' accuracy metric '''

  #pylint: disable=useless-super-delegation
  def __init__(self, config):
    super().__init__(config)

  def call(self, y_true=None, y_pred=None, arguments=None):
    ''' compute metric '''
    return metrics.classification_report(y_true, y_pred)


@registers.metric.register
class CrfCal(Metric):
  ''' crf(ner) metric '''

  #pylint: disable=useless-super-delegation
  def __init__(self, config):
    super().__init__(config)

  def call(self, y_true=None, y_pred=None, arguments=None):
    ''' compute metric '''
    label_path_file = self.config["data"]["task"]["label_vocab"]
    return seq_classification_report(ids_to_sentences(y_true, label_path_file),
                                 ids_to_sentences(y_pred, label_path_file))


def get_metrics(config, y_true=None, y_pred=None):
  ''' candies function of metrics
      calc metrics through `y_true` and `y_pred` or
      `confusion` will be deprecated
  '''
  metrics_list_config = config['solver']['metrics']['cals']
  score = dict()
  for metric in metrics_list_config:
    metric_name = metric['name']
    calculator = registers.metric[metric_name](config)
    arguments = metric['arguments']
    metric_score = calculator(y_true=y_true, y_pred=y_pred, arguments=arguments)
    score[metric_name] = metric_score
  return score
