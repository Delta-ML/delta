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
''' multimodal cls model'''

from delta.models.base_model import Model
import delta.layers
import pickle

import delta.compat as tf
from absl import logging

class AlignClassModel(Model):
  def __init__(self, config, **kwargs):
    super().__init__(**kwargs)

    model_config = config['model']['net']['structure']
    self.taskconf = self.config['data']['task']
    self.audioconf = self.taskconf['audio']

    frame_per_sec = 1 / self.taskconf['audio']['winstep']
    self.input_len = self.taskconf['audio']['clip_size'] * frame_per_sec
    self.input_type = 'samples' if self.taskconf[
        'suffix'] == '.wav' else 'features'
    self.input_channels = 3 if self.taskconf['audio']['add_delta_deltas'] else 1

    self.vocab_size = config['data']['vocab_size']
    self.use_pretrained_embedding = config['model']['use_pre_train_emb']
    self.embedding_size = model_config['embedding_size']
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

    self.speech_enc_layer = delta.layers.RnnEncoder(config, name="speech_encoder")
    self.text_enc_layer = delta.layers.RnnEncoder(config, name="text_encoder")

    self.align_attn_layer = delta.layers.MultiHeadAttention(
      self.hidden_dim, self.head_num)

    self.align_enc_layer = delta.layers.RnnAttentionEncoder(
        config, name="align_encoder")

    self.embed_d = tf.keras.layers.Dropout(self.dropout_rate)
    self.speech_enc_d = tf.keras.layers.Dropout(self.dropout_rate)
    self.text_enc_d = tf.keras.layers.Dropout(self.dropout_rate)
    self.attn_enc_d = tf.keras.layers.Dropout(self.dropout_rate)
    self.align_enc_d = tf.keras.layers.Dropout(self.dropout_rate)

    self.final_dense = tf.keras.layers.Dense(
        self.num_classes,
        activation=tf.keras.activations.linear,
        name="final_dense")

  def call(self, inputs, training=None, mask=None):
    speechs = inputs['inputs']
    texts = inputs['texts']

    text_embed = self.embed(texts)
    text_embed = self.embed_d(text_embed)
    text_enc = self.text_enc_layer(text_embed)
    text_enc = self.text_enc_d(text_enc)

    speech_enc = self.speech_enc_layer(speechs)
    speech_enc = self.text_enc_d(speech_enc)

    attn_outs = self.align_attn_layer(text_enc, speech_enc, speech_enc)
    attn_outs = self.attn_enc_d(attn_outs)

    align_enc = self.align_enc_layer(attn_outs)
    align_enc = self.align_enc_d(align_enc)

    scores = self.final_dense(align_enc)

    return scores
