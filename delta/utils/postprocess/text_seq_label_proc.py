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
    output_index = predictions["output_index"]
    if output_index is None:
      res_file = self.config["solver"]["postproc"].get("res_file", "")
      label_path_file = self.config["data"]["task"]["label_vocab"]
    else:
      res_file = self.config["solver"]["postproc"][output_index].get(
          "res_file", "")
      label_path_file = self.config["data"]["task"]["label_vocab"][output_index]

    if res_file == "":
      logging.info(
          "Infer res not saved. You can check 'res_file' in your config.")
      return
    res_dir = os.path.dirname(res_file)
    if not os.path.exists(res_dir):
      os.makedirs(res_dir)
    logging.info("Save inference result to: {}".format(res_file))

    preds = ids_to_sentences(preds, label_path_file)

    with open(res_file, "w", encoding="utf-8") as in_f:
      for i, pre in enumerate(preds):
        entities = get_entities(pre)  # [('PER', 0, 1), ('LOC', 3, 3)]
        if not entities:
          in_f.write("Null")
        else:
          new_line = "\t".join(
              [" ".join(map(str, entity)) for entity in entities])
          in_f.write(new_line)
        in_f.write("\n")
