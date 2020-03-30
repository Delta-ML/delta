from .transformer import *
from .common import *

#tf.compat.v1.disable_eager_execution()
#
#batch_size = 40
#seq_length = 200
#hidden_size = 768
#num_attention_heads =12
#size_per_head = int(hidden_size / num_attention_heads)
#
#layer_input = tf.compat.v1.placeholder(tf.float32, shape=(batch_size*seq_length, hidden_size))
## Tensor of shape [batch_size, from_seq_length, to_seq_length].
#attention_mask = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, seq_length, seq_length))
#
#output_rnn = transformer_cell(input_tensor=layer_input,#tf.reshape(layer_input, [-1, hidden_size]),
#                              attention_mask=attention_mask,
#                              hidden_size=hidden_size,
#                              num_attention_heads=num_attention_heads,
#                              attention_head_size=size_per_head,
#                              batch_size = batch_size,
#                              seq_length = seq_length,
#                              intermediate_size=1280)
