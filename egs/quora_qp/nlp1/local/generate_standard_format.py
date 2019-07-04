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

import csv
import sys
from absl import logging


def generate_standard_format(input_file,output_file):
  with open(input_file, encoding="utf-8") as csv_file, \
    open(output_file, "w", encoding="utf-8") as out_file:
    csv_reader = csv.reader(csv_file,delimiter='\t')
    csv_reader=list(csv_reader)[1:]
    for row in csv_reader:
      if len(row) < 6:
        continue
      label = row[5]
      id=row[0]
      q1=row[3]
      q2=row[4]
      out_file.write(label + "\t" + q1+"\t"+q2+ "\n")
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
