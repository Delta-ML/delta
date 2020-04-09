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
"""Utilities for building transformer layers."""
import math
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

def log_prob_from_logits(logits, reduce_axis=-1):
  """return log prob use log sum func"""
  return logits - tf.reduce_logsumexp(logits, axis=reduce_axis, keepdims=True)

#
def shape_list(tensor):
  """Return list of dims, statically where possible."""
  tensor = tf.convert_to_tensor(tensor)

  if tensor.get_shape().dims is None:
    return tf.shape(tensor)

  static = tensor.get_shape().as_list()
  shape = tf.shape(tensor)

  ret = []
  for i, _ in enumerate(static):
    dim = static[i]
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret


def merge_beam_dim(tensor):
  """Reshapes first two dimensions in to single dimension."""
  shape = shape_list(tensor)
  shape[0] *= shape[1]  # batch -> batch * beam_size
  shape.pop(1)  # Remove beam dim
  return tf.reshape(tensor, shape)


def unmerge_beam_dim(tensor, batch_size, beam_size):
  """Reshapes first dimension back to [batch_size, beam_size]."""
  shape = shape_list(tensor)
  new_shape = [batch_size] + [beam_size] + shape[1:]
  return tf.reshape(tensor, new_shape)


def expand_to_beam_size(tensor, beam_size):
  """Tiles a given tensor by beam_size."""
  tensor = tf.expand_dims(tensor, axis=1)
  tile_dims = [1] * tensor.shape.ndims
  tile_dims[1] = beam_size

  return tf.tile(tensor, tile_dims)


def get_state_shape_invariants(tensor):
  """Returns the shape of the tensor but sets middle dims to None."""
  shape = tensor.shape.as_list()
  for i in range(1, len(shape) - 1):
    shape[i] = None
  return tf.TensorShape(shape)


def compute_batch_indices(batch_size, beam_size):
  """Computes the i'th coordinate that contains the batch index for gathers."""
  batch_pos = tf.range(batch_size * beam_size) // beam_size
  batch_pos = tf.reshape(batch_pos, [batch_size, beam_size])
  return batch_pos


def create_make_unique(inputs):
  """Replaces the lower bits of each element with iota."""
  if inputs.shape.ndims != 2:
    raise ValueError("Input of top_k_with_unique must be rank-2 "
                     "but got: %s" % inputs.shape)

  height = inputs.shape[0]
  width = inputs.shape[1]
  zeros = tf.zeros([height, width], dtype=tf.int32)

  log2_ceiling = int(math.ceil(math.log(int(width), 2)))
  next_power_of_two = 1 << log2_ceiling
  count_mask = ~(next_power_of_two - 1)
  count_mask_r0 = tf.constant(count_mask)
  count_mask_r2 = tf.fill([height, width], count_mask_r0)

  smallest_normal = 1 << 23
  smallest_normal_r0 = tf.constant(smallest_normal, dtype=tf.int32)
  smallest_normal_r2 = tf.fill([height, width], smallest_normal_r0)

  low_bit_mask = ~(1 << 31)
  low_bit_mask_r0 = tf.constant(low_bit_mask, dtype=tf.int32)
  low_bit_mask_r2 = tf.fill([height, width], low_bit_mask_r0)

  iota = tf.tile(
      tf.expand_dims(tf.range(width, dtype=tf.int32), 0), [height, 1])

  input_r2 = tf.bitcast(inputs, tf.int32)
  abs_r2 = tf.bitwise.bitwise_and(input_r2, low_bit_mask_r2)
  if_zero_r2 = tf.equal(abs_r2, zeros)
  smallest_normal_preserving_sign_r2 = tf.bitwise.bitwise_or(
      input_r2, smallest_normal_r2)
  input_no_zeros_r2 = tf.where(if_zero_r2, smallest_normal_preserving_sign_r2,
                               input_r2)

  and_r2 = tf.bitwise.bitwise_and(input_no_zeros_r2, count_mask_r2)
  or_r2 = tf.bitwise.bitwise_or(and_r2, iota)
  return tf.bitcast(or_r2, tf.float32)


def create_topk_unique(inputs, k):
  """Creates the top k values in sorted order with indices."""
  height = inputs.shape[0]
  width = inputs.shape[1]
  neg_inf_r0 = tf.constant(-np.inf, dtype=tf.float32)
  ones = tf.ones([height, width], dtype=tf.float32)
  neg_inf_r2 = ones * neg_inf_r0
  inputs = tf.where(tf.is_nan(inputs), neg_inf_r2, inputs)

  tmp = inputs
  topk_r2 = tf.zeros([height, k], dtype=tf.float32)
  for i in range(k):
    kth_order_statistic = tf.reduce_max(tmp, axis=1, keepdims=True)
    k_mask = tf.tile(
        tf.expand_dims(tf.equal(tf.range(k), tf.fill([k], i)), 0), [height, 1])
    topk_r2 = tf.where(k_mask, tf.tile(kth_order_statistic, [1, k]), topk_r2)
    ge_r2 = tf.greater_equal(inputs, tf.tile(kth_order_statistic, [1, width]))
    tmp = tf.where(ge_r2, neg_inf_r2, inputs)

  log2_ceiling = int(math.ceil(math.log(float(int(width)), 2)))
  next_power_of_two = 1 << log2_ceiling
  count_mask = next_power_of_two - 1
  mask_r0 = tf.constant(count_mask)
  mask_r2 = tf.fill([height, k], mask_r0)
  topk_r2_s32 = tf.bitcast(topk_r2, tf.int32)
  topk_indices_r2 = tf.bitwise.bitwise_and(topk_r2_s32, mask_r2)
  return topk_r2, topk_indices_r2


def top_k_with_unique(inputs, k):
  """Finds the values and indices of the k largests entries."""
  unique_inputs = create_make_unique(tf.cast(inputs, tf.float32))
  top_values, indices = create_topk_unique(unique_inputs, k)
  top_values = tf.cast(top_values, inputs.dtype)
  return top_values, indices


def compute_topk_scores_and_seq(sequences,
                                scores,
                                scores_to_gather,
                                flags,
                                beam_size,
                                batch_size,
                                prefix="default",
                                states_to_gather=None):
  """Given sequences and scores, will gather the top k=beam size sequences."""
  _, topk_indexes = tf.nn.top_k(scores, k=beam_size)
  batch_pos = compute_batch_indices(batch_size, beam_size)
  top_coordinates = tf.stack([batch_pos, topk_indexes], axis=2)

  def gather(tensor, name):
    return tf.gather_nd(tensor, top_coordinates, name=(prefix + name))

  topk_seq = gather(sequences, "_topk_seq")
  topk_flags = gather(flags, "_topk_flags")
  topk_gathered_scores = gather(scores_to_gather, "_topk_scores")
  if states_to_gather:
    topk_gathered_states = nest.map_structure(
        lambda state: gather(state, "_topk_states"), states_to_gather)
  else:
    topk_gathered_states = states_to_gather

  return topk_seq, topk_gathered_scores, topk_flags, topk_gathered_states


def create_padding_mask(seq):
  """
  create padding mask
  """
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
  # seq = tf.expand_dims(seq, 1)
  # seq = tf.expand_dims(seq, 1)
  # return seq


def create_masks(inp, tar):
  """
  create encode and decode mask
  """
  # encoder self-attention mask
  enc_padding_mask = create_padding_mask(inp)
  # decoder self-attention mask
  dec_padding_mask = create_padding_mask(inp)

  # decoder context mask
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

  return enc_padding_mask, combined_mask, dec_padding_mask

def create_look_ahead_mask(tar):
  """
  create look ahead mask for decode mask
  """
  size = tf.shape(tar)[1]
  look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  dec_target_padding_mask = create_padding_mask(tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  return combined_mask  # (batch_size, 1, seq_len, seq_len)
