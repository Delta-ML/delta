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
''' ctc tensorflow decoder '''

import delta.compat as tf
from delta.utils import ctc_utils


def ctc_decode_blankid_to_last(logits, sequence_length, blank_id=None):
  '''
    Moves the blank_label cloumn to the end of the logit matrix,
    and adjust the rank of sequence_length to 1
    param: logits, (B, T, C), output of ctc asr model
    param: sequence_length, (B, 1), sequence lengths
    param: blank_id, None, default blank_id is 0, same to espnet.
    return: logits_return, (T, B, C)
            sequence_length_return, (B)
    '''

  logits = tf.transpose(logits, [1, 0, 2])
  #blank_id=0 is used as default in Espnet,
  #while blank_id is set as C-1 in tf.nn.ctc_decoder
  if blank_id is None:
    blank_id = 0
  logits = ctc_utils.logits_blankid_to_last(logits=logits, blank_index=blank_id)

  sequence_length_return = tf.cond(
      pred=tf.equal(tf.rank(sequence_length), 1),
      true_fn=lambda: sequence_length,
      false_fn=lambda: tf.squeeze(sequence_length),
  )

  return logits, sequence_length_return, blank_id


def ctc_greedy_decode(logits,
                      sequence_length,
                      merge_repeated=True,
                      blank_id=None):
  '''
    ctc greedy decode function
    param: logits, (B, T, C), output of ctc asr model
    param: sequence_length, (B, 1), sequence lengths
    param: merge_repeated, boolean, if merge consecutive repeated classes in output
    returns:
        decode_result, (B, T), decode result
        probs, (B), A float matrix containing, for the sequence found,
            the negative of the sum of the greatest logit at each timeframe.
    '''

  logits, sequence_len, blank_id = ctc_decode_blankid_to_last(
      logits, sequence_length, blank_id)
  deps = [
      tf.assert_rank(logits, 3),
      tf.assert_rank(sequence_len, 1),
  ]

  with tf.control_dependencies(deps):
    decode_result, probs = tf.nn.ctc_greedy_decoder(
        logits, sequence_len, merge_repeated=merge_repeated)
    decode_result = [
        ctc_utils.labels_last_to_blankid(single_decode_result, blank_id)
        for single_decode_result in decode_result
    ]
    decode_result = tf.sparse_tensor_to_dense(
        decode_result[0], default_value=blank_id, name="outputs")
  return decode_result, probs


def ctc_beam_search_decode(logits,
                           sequence_length,
                           beam_width=1,
                           top_paths=1,
                           blank_id=None):
  '''
    ctc beam search decode function
    param: logits, (B, T, C), output of ctc asr model
    param: sequence_length, (B, 1), sequence lengths
    param: beam_width, int, beam search beam width
    param: top_paths, int, controls output size
    return:
       decode_result, (B, T), decode result
       probs: A float matrix [batch_size, top_paths] containing sequence log-probabilities.
    '''

  logits, sequence_len, blank_id = ctc_decode_blankid_to_last(
      logits, sequence_length, blank_id)

  deps = [tf.assert_rank(logits, 3), tf.assert_rank(sequence_len, 1)]

  with tf.control_dependencies(deps):
    decode_result, probs = tf.nn.ctc_beam_search_decoder_v2(
        logits, sequence_len, beam_width=beam_width, top_paths=top_paths)
    decode_result_recovery_blank_id = [
        ctc_utils.labels_last_to_blankid(single_decode_result, blank_id)
        for single_decode_result in decode_result
    ]
    decode_result_dense = [
        tf.sparse_tensor_to_dense(result, default_value=blank_id)
        for result in decode_result_recovery_blank_id
    ]
  return decode_result_dense, probs
