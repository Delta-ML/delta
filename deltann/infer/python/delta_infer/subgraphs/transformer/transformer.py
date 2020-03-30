from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import numpy as np
import tensorflow as tf
from ..common import *
from .model import *

__all__ = ["transformer_cell"]

@RegistPattern(name="TransformerCell")
def transformer_cell(input_tensor,
                     attention_mask=None,
                     hidden_size=768,
                     num_attention_heads=12,
                     attention_head_size=0,
                     batch_size=0,
                     seq_length=0,
                     intermediate_size=3072,
                     intermediate_act_fn=gelu,
                     hidden_dropout_prob=0.1,
                     attention_probs_dropout_prob=0.1,
                     initializer_range=0.02):
    """Multi-headed, multi-layer Transformer from "Attention is All You Need".
       code ref: https://github.com/google-research/bert/blob/master/modeling.py
    """
    with tf.variable_scope("attention"):
        attention_heads = []
        with tf.variable_scope("self"):
            attention_head = attention_layer(
              from_tensor=input_tensor,
              to_tensor=input_tensor,
              attention_mask=attention_mask,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              attention_probs_dropout_prob=attention_probs_dropout_prob,
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
        # with `input_tensor`.
        with tf.variable_scope("output"):
          attention_output = tf.layers.dense(
              attention_output,
              hidden_size,
              kernel_initializer=create_initializer(initializer_range))
          attention_output = dropout(attention_output, hidden_dropout_prob)
          attention_output = layer_norm(attention_output + input_tensor)

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
        layer_output = dropout(layer_output, hidden_dropout_prob)
        layer_output = layer_norm(layer_output + attention_output)
    return layer_output
