import tensorflow as tf
import delta_infer as dti
from standard_transformer.model import *

@dti.RegistPattern(name="TransformerCell")
def standard_transformer(input_tensor = None,
                         attention_mask = None,
                         hidden_size = None,
                         num_hidden_layers = 1,
                         num_attention_heads = 12,
                         intermediate_size = 12802,
                         intermediate_act_fn = gelu,
                         hidden_dropout_prob = 0.1,
                         initializer_range = 0.02,
                         batch_size=None,
                         seq_length=None,
                         attention_head_size=None):
    with tf.variable_scope("layer_0"):
        layer_input = input_tensor
        with tf.variable_scope("attention"):
            attention_heads = []
            with tf.variable_scope("self"):
                attention_head = attention_layer(
                        from_tensor=layer_input,
                        to_tensor=layer_input,
                        attention_mask=attention_mask,
                        num_attention_heads=num_attention_heads,
                        size_per_head=attention_head_size,
                        attention_probs_dropout_prob=hidden_dropout_prob,
                        initializer_range=initializer_range,
                        do_return_2d_tensor=True,
                        batch_size=batch_size,
                        from_seq_length=seq_length,
                        to_seq_length=seq_length)
                attention_heads.append(attention_head)
            attention_output = None
            if len(attention_heads) == 1:
                attention_output = attention_heads[0]
            else:
                # In the case where we have other sequences, we just concatenate
                # them to the self-attention head before the projection. 
                attention_output = tf.concat(attention_heads, axis=-1)

            # Run a linear projection of `hidden_size` then add a residual
            # with `layer_input`.
            with tf.variable_scope("output"):
                attention_output = tf.layers.dense(
                                    attention_output,
                                    hidden_size,
                                    kernel_initializer=create_initializer(initializer_range))
                #attention_output = dropout(attention_output, hidden_dropout_prob)
                attention_output = layer_norm(attention_output + layer_input)

        # The activation is only applied to the "intermediate" hidden layer. 
        with tf.variable_scope("intermediate"):
            intermediate_output = tf.layers.dense(
                    attention_output,
                    intermediate_size,
                    activation=intermediate_act_fn,
                    kernel_initializer=create_initializer(initializer_range))

        # Down-project back to `hidden_size` then add the residual.
        with tf.variable_scope("output"):
            layer_output = tf.layers.dense(
                    intermediate_output,
                    hidden_size,
                    kernel_initializer=create_initializer(initializer_range))
            #layer_output = dropout(layer_output, hidden_dropout_prob)
            layer_output = layer_norm(layer_output + attention_output)
    return layer_output

if __name__ == "__main__":
    # open graph optimizer stream
    with dti.GraphStream("/path/to/DeepLearningExamples/FasterTransformer/build/nv_fasttransformer.pb") as gs:
        batch_size = 1
        seq_length = 100
        hidden_size = 768
        num_attention_heads =12
        attention_head_size = int(hidden_size / num_attention_heads)

        # Tensor of shape [batch_size, from_seq_length, to_seq_length].
        attention_mask = tf.placeholder(tf.int32, shape=(batch_size, seq_length, seq_length))

        layer_input = tf.placeholder(tf.float32, shape=(batch_size * seq_length, hidden_size))

        output_rnn = standard_transformer(input_tensor=layer_input,
                                          attention_mask=attention_mask,
                                          hidden_size=hidden_size,
                                          num_attention_heads=num_attention_heads,
                                          intermediate_size=1280,
                                          batch_size=batch_size,
                                          seq_length=seq_length,
                                          attention_head_size=attention_head_size)

        # remove in the future
        gs.register_hint_op("TransformerCell", "BatchMatMulV2")
        gs.save("./result.pb")

    with tf.compat.v1.Session() as sess:
        graph_def = dti.RegistPattern.get_patterns("TransformerCell")[0]
        with open("TransformerCell.pb", "wb") as f:
            f.write(graph_def.SerializeToString())

