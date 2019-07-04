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
''' metrics utils of numpy '''
import numpy as np

#pylint: disable=invalid-name
def f1_score(tn, fp, fn, tp):
  '''  f1 score from confusion matrix '''
  del tn
  return 2 * tp / (2 * tp + fp + fn)


#pylint: disable=invalid-name
def stats_confusion(confusion):
  '''' confusion matrix to TP, FP, TN, FN '''
  FP = confusion.sum(axis=0) - np.diag(confusion)
  FN = confusion.sum(axis=1) - np.diag(confusion)
  TP = np.diag(confusion)
  TN = confusion.sum() - (TP + FN + TP)
  return TN, FP, FN, TP
