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

#!/usr/bin/env python
import numpy as np
import os
import sys
import pprint
from sklearn.metrics import recall_score as recall
from sklearn.metrics import confusion_matrix as confusion

import post_process as post_lib
import config

path = sys.argv[1]

# order metrics
order = dict()
with open(path, 'r') as f:
  for line in f:
    filepath, cnt = line.split()
    orderid = os.path.split(filepath)[-1].split('_')[0]

    if config.pos_class in filepath:
      label = 1  # conflict
    else:
      label = 0  # normal

    if orderid in order:
      # if has one segment predict True, then the Order be True
      if order[orderid][1] == 0 and label == 1:
        order[orderid][1] = label
      elif label == 0 and order[orderid][1] == 1:
        pass
      elif order[orderid][1] == 0 and label == 0:
        pass
      elif label == 1 and order[orderid][1] == 1:
        pass
      else:
        raise ValueError("erro")

      order[orderid][0] += int(cnt)
      order[orderid][-1].append(filepath + ':' + cnt)
    else:
      order[orderid] = [int(cnt), label, [filepath + ':' + cnt]]

pprint.pprint(order)

y_true = [v[1] for k, v in order.items()]
y_pred = [v[0] for k, v in order.items()]

y_pred = np.int_(np.array(y_pred) > 0)

conf = confusion(y_true=y_true, y_pred=y_pred, labels=[0, 1])

tn, fp, fn, tp = conf.ravel()
print('tp', tp, 'tn', tn, 'fp', fp, 'fn', fn)
valid_recall = tp / (tp + fn)
valid_prec = tp / (tp + fp)
f1 = post_lib.f1_score(tn, fp, fn, tp)
print("recall: {}".format(valid_recall))
print("precision: {}".format(valid_prec))
print("f1: {}".format(f1))
print('Confusion Matrix: {}'.format(config.class_vocab))
print(conf)
