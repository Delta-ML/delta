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
''' emotion keras model'''
import numpy as np
from absl import logging

#pylint: disable=no-name-in-module
import delta.compat as tf
import delta.layers
from delta.models.base_model import Model

from delta.utils.register import registers
from delta.layers.utils import get_expand_pad_mask
import pickle


@registers.model.register
class AlignClassModel(Model):

  def __init__(self, config, **kwargs):
    super().__init__(**kwargs)
    self.config = config
    config = self.config
    model_config = config['model']['net']['structure']
    self.num_classes = config['data']['task']['classes']['num']
    self.vocab_size = config['data']['task']['text']['vocab_size']
    self.max_text_len = config['data']['task']['text']['max_text_len']
    self.use_pretrained_embedding = config['model']['use_pre_train_emb']
    self.embedding_size = model_config['embedding_size']
    self.hidden_dim = model_config['hidden_dim']
    self.head_num = model_config['head_num']
    self.inner_size = model_config['inner_size']
    self.dropout_rate = model_config['dropout_rate']
    self.speech_dropout_rate = model_config['speech_dropout_rate']
    self.padding_id = model_config.get('padding_id', 0)
    self.speech_dense_act = config.get('speech_dense_act', 'relu')

    if self.use_pretrained_embedding:
      self.embedding_path = config['model']['embedding_path']
      logging.info("Loading embedding file from: {}".format(
          self.embedding_path))
      self._word_embedding_init = pickle.load(open(self.embedding_path, 'rb'))
      self.embed_initializer = tf.constant_initializer(
          self._word_embedding_init)
    else:
      self.embed_initializer = tf.random_uniform_initializer(-0.1, 0.1)

    self.embed = tf.keras.layers.Embedding(
        self.vocab_size,
        self.embedding_size,
        embeddings_initializer=self.embed_initializer)
    self.speech_enc_layer = delta.layers.RnnEncoder(
        config, name="speech_encoder")
    self.text_enc_layer = delta.layers.RnnEncoder(config, name="text_encoder")

    self.align_attn_layer = delta.layers.MultiHeadAttention(
        self.hidden_dim, self.head_num)
    self.align_enc_layer = delta.layers.RnnAttentionEncoder(
        config, name="align_encoder")

    self.embed_d = tf.keras.layers.Dropout(self.dropout_rate)
    self.speech_d = tf.keras.layers.Dropout(self.speech_dropout_rate)
    self.speech_enc_d = tf.keras.layers.Dropout(self.speech_dropout_rate)
    self.text_enc_d = tf.keras.layers.Dropout(self.dropout_rate)
    self.attn_enc_d = tf.keras.layers.Dropout(self.dropout_rate)
    self.align_enc_d = tf.keras.layers.Dropout(self.dropout_rate)
    self.final_dense = tf.keras.layers.Dense(
        self.num_classes, activation=tf.keras.activations.linear)

    self.align_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.speech_dense = tf.keras.layers.Dense(
        512, activation=self.speech_dense_act)

  def build(self, input_shape):
    logging.info(f"{self.__class__.__name__} input_shape : {input_shape}")
    _, time, feat, channels = input_shape['inputs'].as_list()
    self.speech = tf.keras.layers.InputLayer(
        input_shape=(None, time, feat, channels), name="speech")
    self.text = tf.keras.layers.InputLayer(
        input_shape=(None, self.max_text_len), name="text")
    self.reshape1 = tf.keras.layers.Reshape((time, feat * channels),
                                            input_shape=(time, feat, channels))
    self.concatenate = tf.keras.layers.Concatenate()
    self.reshape2 = tf.keras.layers.Reshape(
        (self.max_text_len, self.hidden_dim), input_shape=(-1, self.hidden_dim))
    self.built = True

  def call(self, inputs, training=None, mask=None):
    #logging.info(f"xxxx input: {inputs}, training: {training}")
    speechs = inputs['inputs']
    speechs = self.speech(speechs)
    texts = inputs['texts']
    texts = self.text(texts)
    speechs = self.reshape1(speechs)
    speechs = self.speech_dense(speechs)
    speechs = self.speech_d(speechs)

    texts_mask = get_expand_pad_mask(texts, self.padding_id)

    text_embed = self.embed(texts)
    text_embed = self.embed_d(text_embed, training=training)

    text_enc, _ = self.text_enc_layer(
        text_embed, training=training, mask=texts_mask)

    text_enc = self.text_enc_d(text_enc, training=training)

    speech_enc, _ = self.speech_enc_layer(speechs, training=training)
    speech_enc = self.speech_enc_d(speech_enc, training=training)

    attn_outs, _ = self.align_attn_layer((text_enc, speech_enc, speech_enc),
                                         training=training)
    attn_outs = self.attn_enc_d(attn_outs, training=training)
    attn_outs = self.reshape2(attn_outs)

    attn_outs = self.concatenate([attn_outs, text_enc])
    align_enc = self.align_enc_layer(
        attn_outs, training=training, mask=texts_mask)

    align_enc = self.align_enc_d(align_enc, training=training)

    scores = self.final_dense(align_enc)
    return scores
