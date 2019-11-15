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
"""Plot graphs."""

# pylint: disable=wrong-import-position
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_roc(fpr, tpr, thresholds, auc, save_path):
  ''' thresholds from hight to low '''
  plt.figure()
  plt.plot(
      fpr, tpr, color='darkorange', lw=2, label='ROC curve (area= %0.2f)' % auc)
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic')
  plt.legend(loc='lower right')

  # create the axis of thresholds (scores)
  ax2 = plt.gca().twinx()
  ax2.plot(fpr, thresholds, markeredgecolor='r', linestyle='dashed', color='r')
  ax2.set_ylabel('Threshold', color='r')
  ax2.set_ylim([thresholds[-1], thresholds[0]])
  ax2.set_xlim([fpr[0], fpr[-1]])
  plt.savefig(save_path)
  plt.close()


def plot_pr(precision, recall, thresholds, save_path):  # pylint: disable=unused-argument
  ''' thresholds from low to hight '''
  plt.figure()

  plt.step(recall, precision, color='b', alpha=0.2, where='post')
  plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

  plt.ylim([0.0, 1.05])
  plt.xlim([0.0, 1.0])
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title('2-class Precision-Recall curve')
  plt.legend(loc='lower right')

  # pylint: disable=pointless-string-statement
  '''
    # create the axis of thresholds (scores)
    thresholds = np.append(thresholds, 1.0)
    ax2 = plt.gca().twinx()
    ax2.plot(recall, thresholds, markeredgecolor='r', linestyle='dashed', color='r')
    ax2.set_ylabel('Threshold', color='r')
    ax2.set_ylim( [thresholds[0], thresholds[-1]] )
    ax2.set_xlim( [recall[0], recall[-1]] )
  '''

  plt.savefig(save_path)
  plt.close()


def plot_det(fps, fns, save_path):
  """
    This might be a good metric when you are doing a binary classification problem
      with two species and you care how it is incorrectly classifying both species.
    Given false positive and false negative rates,
      produce a DET(detection error rate) Curve.
    The false positive rate is assumed to be increasing while the false
    negative rate is assumed to be decreasing.
  """
  # pylint: disable=unused-argument, unused-variable, unused-variable, invalid-name
  axis_min = min(fps[0], fns[-1])
  fig, ax = plt.subplots()
  ax.plot(fps, fns)
  ax.yscale('log')
  ax.xscale('log')
  ticks_to_use = [
      0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50
  ]
  ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
  ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
  ax.set_xticks(ticks_to_use)
  ax.set_yticks(ticks_to_use)
  ax.axis([0.001, 50, 0.001, 50])
  ax.savefig(save_path)
