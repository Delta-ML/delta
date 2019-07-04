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

import sys
import tensorflow as tf
from sklearn import metrics

import utils


class tf_roc():

  def __init__(self, predict_label_file, true_label_id, threshold_num,
               save_dir):
    '''
        file format: dataid,predict_score,label
        Args:
          predict_score should be between [0, 1]
          label should be {0 , 1}
          threshold_num: number of threshold will plot
        '''
    self.threshold_num = threshold_num
    self.true_label_id = true_label_id  # True exmaple label id, 0 or 1

    self.trues = 0  #total of True labels
    self.fpr = []  #false positive
    self.tpr = []  #true positive
    self.ths = []  #thresholds

    self.save_dir = save_dir
    self.writer = tf.summary.FileWriter(self.save_dir)

    #load predict_label_file to predicts and labels
    self.predicts = []
    self.labels = []
    self.total = 0
    fd = open(predict_label_file)
    fdl = fd.readline()
    while len(fdl) > 0:
      val = fdl.split(',')  # comma seperate
      self.predicts.append(float(val[-1]))  # postive score
      self.labels.append(
          True if int(val[-2]) == self.true_label_id else False)  # True label
      fdl = fd.readline()
      self.total += 1
    fd.close()

  def roc_pr(self):
    '''
        tn, fp, fn, tp = metrics.confusion_matrix(self.labels, self.predicts)
        fpr = tp / (fp + tn)
        fnr = fn / (fn + tp)
        utils.plot_det(fpr, fnr)
        '''

    # using True to replace POS exmaple
    # ROC curve
    self.fpr, self.tpr, self.ths = metrics.roc_curve(
        y_true=self.labels, y_score=self.predicts, pos_label=1)
    print('fpr', self.fpr)
    print('tpr', self.tpr)
    print('ths', self.ths)
    self.auc = metrics.auc(self.fpr, self.tpr)
    print('AUC', self.auc)

    utils.plot_roc(
        self.fpr, self.tpr, self.ths, self.auc, save_path='output/roc.png')

    # PR-curve
    self.precision, self.recall, thresholds = metrics.precision_recall_curve(
        y_true=self.labels, probas_pred=self.predicts, pos_label=1)
    print("precision len", len(self.precision))
    print("recall len", len(self.recall))
    print("thresholds len", len(thresholds))
    utils.plot_pr(
        self.precision, self.recall, thresholds, save_path='output/pr.png')

  def calc(self):
    for label in self.labels:
      if label:
        self.trues += 1
    threshold_step = 1. / self.threshold_num
    for t in range(self.threshold_num + 1):
      th = 1 - threshold_step * t  # from high to low
      tn, tp, fp, fpr, tpr = self._calc_once(th)
      self.fpr.append(fpr)
      self.tpr.append(tpr)
      self.ths.append(th)
      self._save(fpr, tpr)

  def _save(self, fpr, tpr):
    summt = tf.Summary()
    summt.value.add(tag="roc", simple_value=tpr)
    self.writer.add_summary(summt, fpr * 100)  #for tensorboard step drawable
    self.writer.flush()

  def _calc_once(self, t):
    fp = 0
    tp = 0
    tn = 0

    for i in range(self.total):
      if not self.labels[i]:  # label False
        if self.predicts[i] >= t:  # predict Postive
          fp += 1
        else:  # predict Negtive
          tn += 1
      elif self.predicts[i] >= t:  # label True
        tp += 1  # predict Postive

    print(tp, fp, tn)

    tpr = tp / float(self.trues)
    #fpr = fp / float(fp + tn) #precision
    fpr = fp / float(fp + tp)  #detection
    return tn, tp, fp, fpr, tpr


if __name__ == '__main__':
  predict_label_file = sys.argv[1]
  save_dir = 'ckpt'
  true_label_id = 1
  threshold_num = 20

  roc = tf_roc(predict_label_file, true_label_id, int(threshold_num), save_dir)
  roc.roc_pr()
