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

import tensorflow as tf
from absl import logging

#pylint: disable=invalid-name

path = tf.compat.v1.resource_loader.get_path_to_datafile('x_ops.so')
logging.info('x_ops.so path:{}'.format(path))

gen_x_ops = tf.load_op_library(
    tf.compat.v1.resource_loader.get_path_to_datafile('x_ops.so'))

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
jieba_cut = gen_x_ops.jieba_cut
sentence_to_ids = gen_x_ops.sentence_to_ids
delta_delta = gen_x_ops.delta_delta
