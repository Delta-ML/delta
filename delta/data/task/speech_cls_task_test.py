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
''' emotion speech task unittest'''
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import delta.compat as tf
from absl import logging

from delta import utils
from delta.utils.register import registers
from delta.utils.register import import_all_modules_for_register
from delta import PACKAGE_ROOT_DIR


class SpeechClsTaskTest(tf.test.TestCase):
  ''' emotion task test'''

  def setUp(self):
    super().setUp()
    self.conf_str = '''
    data:
      train:
        paths:
        - null 
        segments: null
      eval:
        paths:
        - null
        segments: null
      infer:
        paths:
        - null 
        segments: null
      task:
        name: SpeechClsTask
        suffix: .npy # file suffix
        audio:
          dry_run: false # not save feat
          # params
          clip_size: 30 # clip len in seconds
          stride: 0.5 # stride in ratio of clip_size
          sr: 8000 # sample rate
          winlen: 0.025 # window len
          winstep: 0.01 # window stride
          nfft: 512 # fft bins, default: 512
          lowfreq: 0
          highfreq: null # default: null, 200 points for 800 nfft, 400 points for 1600 nfft
          preemph: 0.97 # default: 0.97
          # extractor
          feature_extractor: tffeat # `tffeat` to use TF feature_extraction .so library, 'pyfeat' to python_speech_feature
          feature_name: fbank # fbank or spec
          save_feat_path: null  # null for dump feat with same dir of wavs
          feature_size: 40 # extract feature size
          add_delta_deltas: true # delta deltas
          # log pwoer
          log_powspec: false # true, save log power spec; otherwise save power spec
          # cmvn
          cmvn: true # apply cmvn or generate cmvn
          cmvn_path: ./cmvn_conflict.npy # cmvn file
        text:
          enable: False
          vocab_path: /vocab/chars5004_attention.txt
          vocab_size: 5004 # vocab size
          max_text_len: 100 # max length for text
        classes:
          num: 2
          vocab:
            normal: 0
            conflict: 1
        num_parallel_calls: 12
        num_prefetch_batch: 2
        shuffle_buffer_size: 200000
        need_shuffle: true
    solver:
      name: EmotionSolver
      optimizer:
        name: adam
        epochs: 5 # maximum epochs
        batch_size: 32 # number of elements in a training batch
        loss: CrossEntropyLoss
        label_smoothing: 0.0 # label smoothing rate
        learning_rate:
          rate: 0.0001 # learning rate of Adam optimizer
          type:  exp_decay # learning rate type
          decay_rate: 0.99  # the lr decay rate
          decay_steps: 100  # the lr decay_step for optimizer
        clip_global_norm: 3.0 # clip global norm
        multitask: False # whether is multi-task
      metrics:
        pos_label: 1 # int, same to sklearn
        cals:
        - name: AccuracyCal
          arguments: null 
        - name: ConfusionMatrixCal
          arguments: null
        - name: PrecisionCal
          arguments:
            average: 'binary'
        - name: RecallCal
          arguments:
            average: 'binary'
        - name: F1ScoreCal
          arguments:
            average: 'binary'
      saver:
        model_path: "ckpt/emotion-speech-cls/test"
        max_to_keep: 10
        save_checkpoints_steps: 100
        keep_checkpoint_every_n_hours: 10000
        checkpoint_every: 100 # the step to save checkpoint
        summary: false
        save_summary_steps: 100
        eval_on_dev_every_secs: 1
        print_every: 10
        resume_model_path: ""
    '''
    import_all_modules_for_register()
    #tempdir = tempfile.mkdtemp()
    tempdir = self.get_temp_dir()

    config_path = str(Path(tempdir).joinpath("speech_task.yaml"))
    logging.info("config path: {}".format(config_path))
    with open(config_path, 'w', encoding='utf-8') as f:  #pylint: disable=invalid-name
      f.write(self.conf_str)

    dataset_path = Path(tempdir).joinpath("data")
    if not dataset_path.exists():
      dataset_path.mkdir()
    postive_path = dataset_path.joinpath("conflict")
    if not postive_path.exists():
      postive_path.mkdir()
    negtive_path = dataset_path.joinpath("normal")
    if not negtive_path.exists():
      negtive_path.mkdir()

    wav_path = Path(PACKAGE_ROOT_DIR).joinpath(
        'data/feat/python_speech_features/english.wav')
    for i in range(10):
      pos_file = postive_path.joinpath("{}.wav".format(i))
      neg_file = negtive_path.joinpath("{}.wav".format(i))
      shutil.copyfile(str(wav_path), str(pos_file))
      shutil.copyfile(str(wav_path), str(neg_file))

    config = utils.load_config(config_path)
    config['data']['train']['paths'] = [str(dataset_path)]
    config['data']['eval']['paths'] = [str(dataset_path)]
    config['data']['infer']['paths'] = [str(dataset_path)]
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
    self.config['data']['task']['suffix'] = '.wav'
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
    segments = self.config['data'][utils.TRAIN]['segments']
    self.config['data'][utils.INFER]['paths'] = paths
    self.config['data'][utils.INFER]['segments'] = segments

    task = self.task_class(self.config, utils.INFER)
    task.generate_cmvn(dry_run=False)

    self.assertTrue(
        os.path.exists(self.config['data']['task']['audio']['cmvn_path']))
    cmvn = np.load(self.config['data']['task']['audio']['cmvn_path'])
    self.assertEqual(cmvn.ndim, 4)

  def test_generate_data(self):
    ''' test generate data'''
    self.config['data']['task']['suffix'] = '.wav'
    task = self.task_class(self.config, utils.TRAIN)

    for inputs, texts, label, filename, clip_id, soft_labels in task.generate_data(
    ):
      if self.config['data']['task']['classes']['positive'] in filename:
        logging.info(
            "feat shape:{} \ntext: {} \nlabels:{} \nfilename:{} \nclip_id:{}\nsoft_labels:{}"
            .format(inputs.shape, texts, label, filename, clip_id, soft_labels))
        break

  #pylint: disable=too-many-locals
  def test_dataset(self):
    ''' dataset unittest'''
    batch_size = 4
    self.config['data']['task']['suffix'] = '.wav'
    self.config['solver']['optimizer']['batch_size'] = batch_size

    task = self.task_class(self.config, utils.TRAIN)

    dataset = task.input_fn(utils.TRAIN, batch_size, 1)()

    features, labels = dataset.make_one_shot_iterator().get_next()
    samples = features['inputs']
    filenames = features['filepath']
    clip_ids = features['clipid']
    soft_labels = features['soft_labels']

    with self.cached_session(use_gpu=False, force_gpu=False) as sess:
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
