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
''' Speech Postprocess '''
import os
import pickle
import collections
import numpy as np
from absl import logging
from sklearn.metrics import confusion_matrix

from delta import utils
from delta.utils.postprocess.base_postproc import PostProc
from delta.utils.register import registers


#pylint: disable=too-many-instance-attributes
@registers.postprocess.register
class EmoPostProc(PostProc):
  ''' emotion postprocess'''

  def __init__(self, config):
    super().__init__(config)

    taskconf = self.config['data']['task']
    self.positive_id = self.config['solver']['metrics']['pos_label']
    self.class_vocab = taskconf['classes']['vocab']
    self.reverse_vocab = taskconf['classes']['reverse_vocab']
    self.num_class = taskconf['classes']['num']

    postconf = self.config['solver']['postproc']
    output_dir = postconf['pred_path']
    self.output_dir = output_dir if output_dir else os.path.join(
        self.config['solver']['saver']['model_path'], 'infer')
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    self.log_verbose = postconf['log_verbose']

    self.eval = postconf['eval']
    self.infer = postconf['infer']

    self.thresholds = postconf['thresholds']
    self.smoothing = postconf['smoothing']['enable']
    self.smoothing_cnt = postconf['smoothing']['count']

    if self.infer:
      self.pred_score_path = os.path.join(self.output_dir, 'predict_scores.txt')
      self.pred_result_path = os.path.join(self.output_dir,
                                           'predict_results.pkl')
    if self.eval:
      self.pred_metrics_path = os.path.join(self.output_dir, 'metrics.txt')

  #pylint: disable=no-self-use
  def update_stats(self, prediction, stats):
    ''' update true label and pred label'''
    y_true = prediction['labels']
    y_pred = np.argmax(prediction['softmax'], -1)
    stats[0] = np.append(stats[0], y_true)
    stats[1] = np.append(stats[1], y_pred)
    return stats

  def update_metrics(self, prediction, confusion):
    ''' update confusion'''
    conf = confusion_matrix(
        y_true=prediction['labels'],
        y_pred=np.argmax(prediction['softmax'], -1),
        labels=list(range(self.num_class)))
    if conf.shape != (self.num_class, self.num_class):
      raise ValueError('Warning: confusion matrix shape error, shape {}'.format(
          conf.shape))
    confusion += conf
    return confusion

  def log_metrics(self, stats):
    ''' compute metrics'''
    socres = utils.metrics.get_metrics(
        self.config, y_true=stats[0], y_pred=stats[1])
    with open(self.pred_metrics_path, 'w') as f:  #pylint: disable=invalid-name
      for key, val in socres.items():
        logging.info("{}: {}".format(key, val))
        f.write("{}: {}\n".format(key, val))

  def collect_results(self, prediction, result, scorefile):
    ''' collect scores and meta of results'''
    for filepath, clipid, label, score in zip(prediction['filepath'],
                                              prediction['clipid'],
                                              prediction['labels'],
                                              prediction['softmax']):

      # record predict score for ROC curve
      scorefile.write("{}, {}, {}, {}\n".format(filepath, clipid, label,
                                                score[self.positive_id]))
      result[filepath].append({
          "clipid": clipid,
          'label': label,
          'softmax': score
      })

  #pylint: disable=too-many-locals,too-many-locals
  def post_proc_results(self, results, thresholds=None):
    ''' smoothing score and predict label '''
    thresholds = thresholds or np.linspace(0, 1, num=10, endpoint=False)
    for threshold in thresholds:
      output_path = os.path.join(self.output_dir,
                                 'predict_ths_%3f.txt' % (threshold))
      with open(output_path, 'w') as predfile:
        # history len `maxlen`
        smoothing = self.smoothing
        maxlen = self.smoothing_cnt
        history = collections.deque(maxlen=maxlen)
        vote_all = collections.deque(maxlen=maxlen)

        for path in results:  # filepath
          # sort file clips by `clipid`
          clips = results[path]
          clips.sort(key=lambda entry: entry['clipid'])
          pos_cnt = 0

          for elem in clips:  # clipid
            clipid = elem['clipid']
            label = elem['label']
            score = elem['softmax']

            # append history deque
            y_pos = score[self.positive_id]
            history.append(y_pos)

            hist = list(history)
            if smoothing:
              # smoothing
              y_pos = np.mean(hist)
            else:
              pass

            # predict result for file
            if y_pos > threshold:
              y_pred = self.positive_id
            else:
              y_pred = 1 - self.positive_id

            if y_pred == self.positive_id:
              pos_cnt += 1
              vote_all.append(True)
            logging.info('file {}, clipid {}, label {}, pred {}'.format(
                path, clipid, label, y_pred))

          #sentence_pred = pos_cnt >= FLAGS.threshold
          sentence_pred = all(list(vote_all))
          sentence_label = self.positive_id if 'conflict' in str(path) else (
              1 - self.positive_id)

          logging.info("file ths_{}: {} {} {}".format(threshold, path, pos_cnt,
                                                      sentence_pred))

          predfile.write("{} {} {}\n".format(path, sentence_label, pos_cnt))

  #pylint: disable=arguments-differ
  def call(self, predictions, log_verbose=False):
    ''' main func entrypoint'''
    # [true_label, pred_label]
    stats = [np.array([]), np.array([])]
    confusion = np.zeros((self.num_class, self.num_class), dtype=np.int32)

    pred_results = collections.defaultdict(list)

    self.pred_score_path = os.path.join(self.output_dir, 'predict_scores.txt')
    self.pred_result_path = os.path.join(self.output_dir, 'predict_results.pkl')

    with open(self.pred_score_path, 'w') as score_file, \
      open(self.pred_result_path, 'wb') as result_file:
      for i, pred in enumerate(predictions):
        if log_verbose or self.log_verbose:
          logging.info('index: {} pred: {}'.format(i, pred))
          for key, val in pred.items():
            logging.info("{}: val: {} type: {}".format(key, val, type(val)))

        if self.eval:
          confusion = self.update_metrics(pred, confusion)
          stats = self.update_stats(pred, stats)

        if self.infer:
          self.collect_results(pred, pred_results, score_file)

      if self.infer:
        pickle.dump(pred_results, result_file)

    if self.eval:
      self.log_metrics(stats)

    if self.infer:
      self.post_proc_results(pred_results, self.thresholds)
