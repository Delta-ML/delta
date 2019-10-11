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
''' Fashion Mnist Task '''
import numpy as np
import tensorflow as tf
from absl import logging

from delta import utils
from delta.utils.hparam import HParams
from delta.utils.register import registers
from delta.data.task.base_task import Task


@registers.task.register
class FashionMnistTask(Task):
  ''' fashion mnist dataset '''

  def __init__(self, config: dict, mode):
    super().__init__(config)
    assert mode in (utils.TRAIN, utils.EVAL, utils.INFER)
    self.mode = mode

  @classmethod
  def params(cls, config: dict = None):
    hp = HParams(cls=cls)
    hp.add_hparam('name', cls.__name__)
    hp.add_hparam('train_buf', 60000)
    hp.add_hparam('test_buf', 10000)
    hp.add_hparam('batch_size', 512)
    hp.add_hparam('need_label', True)
    hp.add_hparam('dims', (28, 28, 1))
    if config:
      hp.override_from_dict(config)
    return hp

  def generate_data(self):
    ''' generate train and test examples(full dataset) '''
    (train_images,
     train_labels), (test_images,
                     test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    logging.info(f"{train_labels[:10]}")
    train_images = train_images.reshape(train_images.shape[0], 28, 28,
                                        1).astype('float32') / 255.0
    test_images = test_images.reshape(test_images.shape[0], 28, 28,
                                      1).astype('float32') / 255.0

    train_labels = train_labels.astype('int32')
    test_labels = test_labels.astype('int32')

    return (train_images, train_labels), (test_images, test_labels)

  def feature_spec(self, batch_size):
    return None

  def preprocess_batch(self, batch):
    return batch

  def dataset(self, batch_size):
    (train_images, train_labels), (test_images,
                                   test_labels) = self.generate_data()

    def _make_ds(images, labels, batch_size, buffer_size, need_label):
      if need_label:
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
      else:
        ds = tf.data.Dataset.from_tensor_slices(images)
      return ds.shuffle(buffer_size).batch(batch_size)

    if self.mode == utils.TRAIN:
      ds = _make_ds(train_images, train_labels, batch_size,
                    self.config.train_buf, self.config.need_label)
    else:
      ds = _make_ds(test_images, test_labels, batch_size, self.config.test_buf,
                    self.config.need_label)

    return ds

  def input_fn(self):
    pass
