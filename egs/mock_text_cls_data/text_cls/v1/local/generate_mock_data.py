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

import sys
from absl import logging
from delta.data.utils.test_utils import mock_a_text_file
from delta.data.utils.test_utils import save_a_vocab_file

samples_english = ["1\tAll is well", "0\tI am very angry"]
samples_split_line_mark = ["1\t都挺好。|都是的呀", "0\t我很愤怒|超级生气！"]
samples_split_by_space = ["1\t都 挺好", "0\t我 很 愤怒"]
samples_split_by_char = ["1\t都挺好", "0\t我很愤怒"]
samples_chinese_word = ["1\t都挺好", "0\t我很愤怒"]

samples_dict = {"english": samples_english,
                "split_by_line_mark": samples_split_line_mark,
                "split_by_space": samples_split_by_space,
                "split_by_char": samples_split_by_char,
                "chinese_word": samples_chinese_word}

text_vocab_english = ["<unk>\t0", "</s>\t1", "all\t3", "is\t4",
                      "well\t5", "i\t6", "am\t7", "very\t8"]
text_vocab_split_line_mark = ["<unk>\t0", "</s>\t1", "都\t2", "挺好\t3",
                              "我\t4", "很\t5", "|\t6", "是的\t7",
                              "呀\t8", "超级\t9", "生气\t10"]
text_vocab_split_by_space = ["<unk>\t0", "</s>\t1", "都\t2", "挺好\t3",
                             "我\t4", "很\t5"]
text_vocab_split_by_char = ["<unk>\t0", "</s>\t1", "都\t2", "挺\t3",
                            "好\t4", "我\t5", "很\t6","愤\t7","怒\t8"]
text_vocab_chinese_word = ["<unk>\t0", "</s>\t1", "都\t2", "挺好\t3",
                             "我\t4", "很\t5"]
text_vocab_dict = {"english": text_vocab_english,
                   "split_by_line_mark": text_vocab_split_line_mark,
                   "split_by_space": text_vocab_split_by_space,
                   "split_by_char": text_vocab_split_by_char,
                   "chinese_word": text_vocab_chinese_word}


def mock_text_class_data(train_file, dev_file, test_file, text_vocab_file, data_type):
  samples = samples_dict[data_type]
  logging.info("Generate mock data: {}".format(train_file))
  mock_a_text_file(samples, 300, train_file)
  logging.info("Generate mock data: {}".format(dev_file))
  mock_a_text_file(samples, 100, dev_file)
  logging.info("Generate mock data: {}".format(test_file))
  mock_a_text_file(samples, 100, test_file)
  text_vocab_list = text_vocab_dict[data_type]
  logging.info("Generate text vocab file: {}".format(text_vocab_file))
  save_a_vocab_file(text_vocab_file, text_vocab_list)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)

  if len(sys.argv) != 5:
    logging.error("Usage python {} train_file dev_file test_file text_vocab_file".format(sys.argv[0]))
    sys.exit(-1)

  for data_type in samples_dict.keys():
    train_file = sys.argv[1].replace("txt", "") + data_type + ".txt"
    dev_file = sys.argv[2].replace("txt", "") + data_type + ".txt"
    test_file = sys.argv[3].replace("txt", "") + data_type + ".txt"
    text_vocab_file = sys.argv[4].replace("txt", "") + data_type + ".txt"

    mock_text_class_data(train_file, dev_file, test_file, text_vocab_file, data_type)
