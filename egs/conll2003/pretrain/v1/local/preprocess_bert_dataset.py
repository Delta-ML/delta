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
from bert import tokenization

def bert_preprocess(filename, vocab):
  tokenizer = tokenization.FullTokenizer(
    vocab_file=vocab, do_lower_case=False)
  new_filename = filename + ".bert"
  f1 = open(new_filename, 'w')
  per_count = 0
  with open(filename, "r") as f:
    lines = f.readlines()
    for line in lines:
      str1 = line.split("\t")[1]
      label1 = line.split("\t")[0]
      new_label_list = []
      old_label_list = label1.split(' ')
      word_list = str1.split(' ')
      tokens = []
      tokens.append('[CLS]')
      new_label_list.append('O')
      per_count = 0
      for i, (w, t) in enumerate(zip(word_list, old_label_list)):
        token = tokenizer.tokenize(w)
        tokens.extend(token)
        for i, _ in enumerate(token):
          if i == 0:
            new_label_list.append(t)
          else:
            new_label_list.append("X")
      tokens.append('[SEG]')
      new_label_list.append('O')
      assert len(tokens) == len(new_label_list)
      rm_new_label_list = [i for i in new_label_list if i != 'O' and i != 'X']
      rm_old_label_list = [i for i in old_label_list if i != 'O' and i != 'X']
      assert len(rm_new_label_list) == len(rm_old_label_list)
      f1.write(" ".join(new_label_list) + '\t' +
               " ".join(tokens) + '\n')

def change_data_format(in_file, out_file):
  with open(out_file, "w") as f_out:
    with open(in_file, "r") as f_in:
      for line in f_in.readlines():
        line_parts = line.strip().split("\t")
        if len(line_parts) != 2:
          logging.error("line error")
        else:
          label = line_parts[0].split(" ")
          text = line_parts[1]
          new_label = [x for x in label]
          ind = 0
          while ind < len(label):
            if label[ind] == "X":
              start, end = ind, ind+1
              while end < len(label) and label[end] == "X":
                end += 1
              label_parts = label[ind-1].split("-")
              if len(label_parts) == 2:
                _, type = label_parts[0], label_parts[1]
                for i in range(start, end):
                  new_label[i] = "I-" + type
              elif len(label_parts) == 1:
                for i in range(start, end):
                  new_label[i] = label_parts[0]
              else:
                logging.error("label error")
              ind = end + 1
            else:
              ind += 1
          assert len(new_label) == len(text.split(" "))
          new_line = " ".join(new_label)+"\t"+text+"\n"
          f_out.write(new_line)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)

  if len(sys.argv) != 5:
    logging.error("Usage python {} train_path, val_path, test_path, raw_vocab"
                  .format(sys.argv[0]))
    sys.exit(-1)

  train_path = sys.argv[1]
  val_path = sys.argv[2]
  test_path = sys.argv[3]
  raw_vocab = sys.argv[4]
  bert_preprocess(train_path, raw_vocab)
  change_data_format(train_path + ".bert", train_path + ".bert.new")
  bert_preprocess(val_path, raw_vocab)
  change_data_format(val_path + ".bert", val_path + ".bert.new")
  bert_preprocess(test_path, raw_vocab)
  change_data_format(test_path + ".bert", test_path + ".bert.new")
