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
''' postprocess utils '''
from absl import logging
from delta.data.preprocess.utils import load_vocab_dict


def ids_to_sentences(ids, vocab_file_path):
  """
  transform array of numbers to array of tags/words
  ids:  [[1,2],[3,4]...]
  """

  vocab_dict = load_vocab_dict(vocab_file_path)
  id_to_vocab = {int(v): k for k, v in vocab_dict.items()}

  sentences = []
  for sent in ids:
    sent_char = []
    for s_char in sent:
      if s_char not in id_to_vocab:
        logging.error("label not in vocabs")
      else:
        sent_char.append(id_to_vocab[s_char])
    sentences.append(sent_char)
  assert len(sentences) == len(ids)
  return sentences
