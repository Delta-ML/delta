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
''' python custom ops '''

import os
import delta.compat as tf
from absl import logging

from delta import PACKAGE_ROOT_DIR
from delta.data.utils import read_lines_from_text_file

#pylint: disable=invalid-name


file_dir = tf.resource_loader.get_data_files_path()
try:
  so_lib_file = tf.io.gfile.glob(file_dir + '/x_ops*.so')[0].split('/')[-1]
except IndexError as e:
  raise FileNotFoundError(f"No x_ops*.so match under dir: {file_dir}")
path = tf.resource_loader.get_path_to_datafile(so_lib_file)

logging.info('x_ops.so path:{}'.format(path))


gen_x_ops = tf.load_op_library(path)

pitch = gen_x_ops.pitch
frame_pow = gen_x_ops.frame_pow
zcr = gen_x_ops.zcr
spectrum = gen_x_ops.spectrum
cepstrum = gen_x_ops.cepstrum
plp = gen_x_ops.plp
analyfiltbank = gen_x_ops.analyfiltbank
synthfiltbank = gen_x_ops.synthfiltbank
fbank = gen_x_ops.fbank
ngram = gen_x_ops.ngram
vocab_token_to_id = gen_x_ops.vocab_token_to_id
vocab_id_to_token = gen_x_ops.vocab_id_to_token
token_in_vocab = gen_x_ops.token_in_vocab
str_lower = gen_x_ops.str_lower
sentence_to_ids = gen_x_ops.sentence_to_ids
delta_delta = gen_x_ops.delta_delta


def jieba_cut(input_sentence,
              use_file=True,
              hmm=True):

  dict_path = os.path.join(PACKAGE_ROOT_DIR,
                           "./resources/cppjieba_dict/jieba.dict.utf8")
  hmm_path = os.path.join(PACKAGE_ROOT_DIR,
                          "./resources/cppjieba_dict/hmm_model.utf8")
  user_dict_path = os.path.join(PACKAGE_ROOT_DIR,
                                "./resources/cppjieba_dict/user.dict.utf8")
  idf_path = os.path.join(PACKAGE_ROOT_DIR,
                          "./resources/cppjieba_dict/idf.utf8")
  stop_word_path = os.path.join(PACKAGE_ROOT_DIR,
                                "./resources/cppjieba_dict/stop_words.utf8")

  if use_file:
    output_sentence = gen_x_ops.jieba_cut(
      input_sentence,
      use_file=use_file,
      hmm=hmm,
      dict_path=dict_path,
      hmm_path=hmm_path,
      user_dict_path=user_dict_path,
      idf_path=idf_path,
      stop_word_path=stop_word_path)
  else:
    dict_lines = read_lines_from_text_file(dict_path)
    model_lines = read_lines_from_text_file(hmm_path)
    user_dict_lines = read_lines_from_text_file(user_dict_path)
    idf_lines = read_lines_from_text_file(idf_path)
    stop_word_lines = read_lines_from_text_file(stop_word_path)

    output_sentence = gen_x_ops.jieba_cut(
      input_sentence,
      use_file=use_file,
      hmm=hmm,
      dict_lines=dict_lines,
      model_lines=model_lines,
      user_dict_lines=user_dict_lines,
      idf_lines=idf_lines,
      stop_word_lines=stop_word_lines)

  return output_sentence
