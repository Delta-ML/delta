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
from absl import logging
from seqeval.metrics.sequence_labeling import get_entities
from delta.utils.postprocess.base_postproc import PostProc
from delta.utils.register import registers
from delta.utils.postprocess.postprocess_utils import ids_to_sentences

#pylint: disable=too-many-instance-attributes, too-few-public-methods, too-many-locals
@registers.postprocess.register
class SavePredEntityPostProc(PostProc):
  '''Save the result of inference.'''

  #pylint: disable=arguments-differ, unused-argument
  def call(self, predictions, log_verbose=False):
    ''' main func entrypoint'''
    preds = predictions["preds"]
    paths = self.config["data"]["infer"]["paths"]
    max_seq_len = self.config["data"]["task"]["max_seq_len"]

    text = []
    counter = 0
    for path in paths:
      with open(path, 'r', encoding='utf8') as file_input:
        for line in file_input.readlines():
          line = list(line.strip())
          if line:
            if len(line) >= max_seq_len:
              line = line[:max_seq_len]
            else:
              line.extend(["unk"]*(max_seq_len-len(line)))
            text.append("".join(line))
            counter += 1
      logging.info("Load {} lines from {}.".format(str(counter), path))

    res_file = self.config["solver"]["postproc"].get("res_file", "")
    if res_file == "":
      logging.info("Infer res not saved. You can check 'res_file' in your config.")
      return
    res_dir = os.path.dirname(res_file)
    if not os.path.exists(res_dir):
      os.makedirs(res_dir)
    logging.info("Save inference result to: {}".format(res_file))
    label_path_file = self.config["data"]["task"]["label_vocab"]
    preds = ids_to_sentences(preds, label_path_file)

    assert len(text) == len(preds)

    with open(res_file, "w", encoding="utf-8") as in_f:
      for i, pre in enumerate(preds):
        entity_dict = {}
        entities = get_entities(pre)  # [('PER', 0, 1), ('LOC', 3, 3)]

        for entity_tuple in entities:
          entity = "".join([text[i][j] for j in range(entity_tuple[1], entity_tuple[2] + 1)])
          if entity_tuple[0] in entity_dict:
            entity_dict[entity_tuple[0]].append(entity)
          else:
            entity_dict[entity_tuple[0]] = [entity]
        in_f.write(str(entity_dict))
        in_f.write("\n")
