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

import json
import sys
from absl import logging


def generate_standard_format(input_file,output_file):
  label_dic={"neutral": "2", "contradiction": "0", "entailment": "1"}
  with open(input_file, encoding="utf-8") as json_file, \
    open(output_file, "w", encoding="utf-8") as out_file:
    text_reader = json_file.readlines()
    for line in text_reader:
      line_dic=json.loads(line)
      if "gold_label" not in line_dic or "sentence1" not in line_dic or "sentence2" not in line_dic:

        continue
      if line_dic["gold_label"]=="-":
        continue
      label = label_dic[line_dic["gold_label"]]
      sentence1 = line_dic["sentence1"]
      sentence2 = line_dic["sentence2"]
      out_file.write(label + "\t" + sentence1 + "\t" + sentence2 + "\n")
if __name__=="__main__":
  logging.set_verbosity(logging.INFO)
  if len(sys.argv) != 3:
    # ['generate_standard_format.py', $data/QQP/original/quora_duplicate_questions.tsv ,$data/quora_stand.txt]
    logging.error("Usage {} input_file output_file".format(sys.argv[0]))
    sys.exit()
  input_file = sys.argv[1]
  output_file = sys.argv[2]
  logging.info("Save file to {}".format(output_file))
  generate_standard_format(input_file,output_file)
