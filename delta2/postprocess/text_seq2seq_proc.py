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
''' Text Sequence to Sequence Postprocess '''
import os
import numpy as np
from absl import logging
from delta.utils.postprocess.base_postproc import PostProc
from delta.utils.register import registers
from delta.utils.postprocess.postprocess_utils import ids_to_sentences

#pylint: disable=too-many-instance-attributes, too-few-public-methods, too-many-locals


@registers.postprocess.register
class SavePredSeqPostProc(PostProc):
  '''Save the result of inference.'''

  #pylint: disable=arguments-differ, unused-argument
  def call(self, predictions, log_verbose=False):
    ''' main func entrypoint'''
    preds = predictions["preds"]

    res_file = self.config["solver"]["postproc"].get("res_file", "")
    if res_file == "":
      logging.info(
          "Infer res not saved. You can check 'res_file' in your config.")
      return
    res_dir = os.path.dirname(res_file)
    if not os.path.exists(res_dir):
      os.makedirs(res_dir)
    logging.info("Save inference result to: {}".format(res_file))
    self.task_config = self.config['data']['task']
    self.label_vocab_file_paths = self.task_config['label_vocab']
    if not isinstance(self.label_vocab_file_paths, list):
      self.label_vocab_file_paths = [self.label_vocab_file_paths]
    self.use_label_vocab = self.task_config['use_label_vocab']
    if self.use_label_vocab:
      label_path_file = self.label_vocab_file_paths[0]
    else:
      label_path_file = self.task_config["text_vocab"]
    preds = ids_to_sentences(preds, label_path_file)
    with open(res_file, "w", encoding="utf-8") as in_f:
      for i, pre in enumerate(preds):
        while len(pre) > 1 and pre[-1] in ['<unk>', '<pad>', '<eos>']:
          pre.pop()
        pred_abs = ' '.join(pre)
        in_f.write(pred_abs)
        in_f.write("\n")
