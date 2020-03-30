import numpy as np
import tensorflow as tf
import delta_infer as dti
from tts_transformer.model import *

def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

class ScaledDotProductAttention(tf.keras.layers.Layer):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights
    """

    def call(self, q, k, v, mask):
        """This is where the layer's logic lives."""
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += mask * -1e9

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        # (..., seq_len_q, seq_len_k)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    """ Multi-head attention

    Multi-head attention consists of four parts: * Linear layers and split into
    heads. * Scaled dot-product attention. * Concatenation of heads. * Final linear layer.
    Each multi-head attention block gets three inputs; Q (query), K (key), V (value).
    These are put through linear (Dense) layers and split up into multiple heads.
    The scaled_dot_product_attention defined above is applied to each head (broadcasted for
    efficiency). An appropriate mask must be used in the attention step. The attention
    output for each head is then concatenated (using tf.transpose, and tf.reshape) and
    put through a final Dense layer.
    Instead of one single attention head, Q, K, and V are split into multiple heads because
    it allows the model to jointly attend to information at different positions from
    different representational spaces. After the split each head has a reduced dimensionality,
    so the total computation cost is the same as a single head attention with full
    dimensionality.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        layers = tf.compat.v1.keras.layers

        self.wq = layers.Dense(
            d_model,
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
            input_shape=(d_model,),
        )
        self.wk = layers.Dense(
            d_model,
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
            input_shape=(d_model,),
        )
        self.wv = layers.Dense(
            d_model,
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
            input_shape=(d_model,),
        )

        self.attention = ScaledDotProductAttention()

        self.dense = layers.Dense(
            d_model,
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
            input_shape=(d_model,),
        )

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).

        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        """ call function """
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, hiddn_dim)
        k = self.wk(k)  # (batch_size, seq_len, hiddn_dim)
        v = self.wv(v)  # (batch_size, seq_len, hiddn_dim)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.attention(q, k, v, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

class TransformerEncoderLayer(tf.keras.layers.Layer):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = tf.random(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        # Implementation of Feedforward model
        #layers = tf.keras.layers
        layers = tf.compat.v1.keras.layers
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(
                    dim_feedforward,
                    activation=gelu,
                    kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                        stddev=0.02
                    ),
                    input_shape=(d_model,),
                ),
                layers.Dropout(dropout, input_shape=(dim_feedforward,)),
                layers.Dense(
                    d_model,
                    kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                        stddev=0.02
                    ),
                    input_shape=(dim_feedforward,),
                ),
                layers.Dropout(dropout, input_shape=(d_model,)),
            ]
        )

        self.norm1 = layers.LayerNormalization(epsilon=1e-8, input_shape=(d_model,))
        self.norm2 = layers.LayerNormalization(epsilon=1e-8, input_shape=(d_model,))
        self.dropout = layers.Dropout(dropout, input_shape=(d_model,))

    def call(self, src, src_mask=None, training=None):
        """Pass the input through the endocder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            mask: the mask for the src sequence (optional).

        Shape:
            see the docs in Transformer class.
        """
        out = self.self_attn(src, src, src, mask=src_mask)[0]
        out = self.norm1(src + self.dropout(out, training=training))
        out = self.norm2(out + self.ffn(out, training=training))

        return out
