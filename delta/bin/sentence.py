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

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import post_process as post_lib
import config

path = sys.argv[1]

# sentence metrics
order = dict()
with open(path, 'r') as f:
  for line in f:
    filepath, cnt = line.split()
    if config.pos_class in filepath:
      label = 1
    else:
      label = 0
    order[filepath] = (int(cnt), label)

pprint.pprint(order)

y_true = [v[1] for k, v in order.items()]
y_pred = [v[0] for k, v in order.items()]

# pos_cnt > 0 then predict True
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
