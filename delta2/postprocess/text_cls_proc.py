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

from delta.utils.postprocess.base_postproc import PostProc
from delta.utils.register import registers


#pylint: disable=too-many-instance-attributes, too-few-public-methods
@registers.postprocess.register
class SavePredPostProc(PostProc):
  '''Save the result of inference.'''

  #pylint: disable=arguments-differ, unused-argument
  def call(self, predictions, log_verbose=False):
    ''' main func entrypoint'''
    logits = predictions["logits"]
    preds = predictions["preds"]
    output_index = predictions["output_index"]
    if output_index is None:
      res_file = self.config["solver"]["postproc"].get("res_file", "")
    else:
      res_file = self.config["solver"]["postproc"][output_index].get(
          "res_file", "")
    if res_file == "":
      logging.info(
          "Infer res not saved. You can check 'res_file' in your config.")
      return
    res_dir = os.path.dirname(res_file)
    if not os.path.exists(res_dir):
      os.makedirs(res_dir)
    logging.info("Save inference result to: {}".format(res_file))
    with open(res_file, "w") as in_f:
      for logit, pred in zip(logits, preds):
        in_f.write(" ".join(["{:.3f}".format(num) for num in logit]) +
                   "\t{}\n".format(pred))
