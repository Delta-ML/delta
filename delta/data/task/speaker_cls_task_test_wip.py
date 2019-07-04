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
''' speaker task unittest'''
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from absl import logging

from delta import utils
from delta.utils.register import registers


class SpeakerClsTaskTest(tf.test.TestCase):
  ''' speaker task test'''

  def setUp(self):
    ''' set up'''
    config_path = 'tests/speaker/speaker_test.yml'

    config = utils.load_config(config_path)
    logging.info("config: {}".format(config))

    solver_name = config['solver']['name']
    self.solver = registers.solver[solver_name](config)

    # config after process
    self.config = self.solver.config

    task_name = self.config['data']['task']['name']
    self.task_class = registers.task[task_name]

  def tearDown(self):
    ''' tear down'''

  def test_generate_feat(self):
    ''' test generate feature'''
    paths = []
    for mode in [utils.TRAIN, utils.EVAL, utils.INFER]:
      paths += self.config['data'][mode]['paths']

    task = self.task_class(self.config, utils.INFER)
    task.generate_feat(paths, dry_run=False)

  def test_generate_cmvn(self):
    ''' test generate cmvn'''
    tmp = tempfile.mktemp(suffix='cmvn.npy')
    self.config['data']['task']['audio']['cmvn_path'] = tmp

    self.config['data']['task']['suffix'] = '.wav'
    self.config['data']['task']['stride'] = 1.0
    paths = self.config['data'][utils.TRAIN]['paths']
    self.config['data'][utils.INFER]['paths'] = paths

    task = self.task_class(self.config, utils.INFER)
    task.generate_cmvn(dry_run=False)

    self.assertTrue(
        os.path.exists(self.config['data']['task']['audio']['cmvn_path']))
    cmvn = np.load(self.config['data']['task']['audio']['cmvn_path'])
    self.assertEqual(cmvn.ndim, 4)

  def test_generate_data(self):
    ''' test generate data'''
    self.config['data']['task']['suffix'] = '.npy'
    task = self.task_class(self.config, utils.TRAIN)

    for inputs, texts, label, filename, clip_id, soft_labels in task.generate_data(
    ):
      logging.info(
          "feat shape:{} \ntext: {} \nlabels:{} \nfilename:{} \nclip_id:{}\nsoft_labels:{}"
          .format(inputs.shape, texts, label, filename, clip_id, soft_labels))
      break

  #pylint: disable=too-many-locals
  def test_dataset(self):
    ''' dataset unittest'''
    batch_size = 4
    self.config['solver']['optimizer']['batch_size'] = batch_size

    task = self.task_class(self.config, utils.TRAIN)

    dataset = task.input_fn(utils.TRAIN, batch_size, 1)()

    features, labels = dataset.make_one_shot_iterator().get_next()
    samples = features['inputs']
    filenames = features['filepath']
    clip_ids = features['clipid']
    soft_labels = features['soft_labels']

    with self.session() as sess:
      while True:
        batch_inputs, batch_labels, batch_files, batch_clipids, labels_onehot, batch_soft_labels = \
           sess.run([samples, labels, filenames, clip_ids, tf.one_hot(labels, 2), soft_labels])

        del labels_onehot
        logging.info("feat shape: {}".format(batch_inputs.shape))
        logging.info("labels: {}".format(batch_labels))
        logging.info("filename: {}".format(batch_files))
        logging.info("clip id: {}".format(batch_clipids))
        logging.info("soft_labels: {}".format(batch_soft_labels))
        break


if __name__ == '__main__':
  logging.set_verbosity(logging.DEBUG)
  tf.test.main()
