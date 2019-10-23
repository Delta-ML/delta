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
import tempfile
from pathlib import Path

import numpy as np
import delta.compat as tf
from absl import logging

from delta import utils
from delta.utils.register import registers
from delta.utils.register import import_all_modules_for_register
from delta.utils.kaldi import kaldi_dir_utils


class SpeakerClsTaskTest(tf.test.TestCase):
  ''' speaker task test'''

  def setUp(self):
    super().setUp()
    import_all_modules_for_register()
    self.conf_str = '''
    data:
      train:
        paths:
        - ''
      eval:
        paths:
        - ''
      infer:
        paths:
        - ''
      task:
        name: SpeakerClsTask
        data_type: KaldiDataDirectory
        suffix: .npy # file suffix
        audio:
          dry_run: false # not save feat
          # params
          clip_size: 3 # clip len in seconds
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
          save_feat_path: null  # null for dump feat with same dir of wavs
          # fbank
          save_fbank: true # save fbank or power spec
          feature_size: 23 # extract feature size
          add_delta_deltas: false # delta deltas
          # log pwoer
          log_powspec: false # true, save log power spec; otherwise save power spec
          # cmvn
          cmvn: true # apply cmvn or generate cmvn
          cmvn_path: ./cmvn_speaker.npy # cmvn file
        classes:
          num: 2 
          vocab: null
        num_parallel_calls: 12
        num_prefetch_batch: 2
        shuffle_buffer_size: 200000
        need_shuffle: true

    model:
      name: SpeakerCRNNRawModel
      type: raw # raw, keras or eager model
      net:
        structure:
          embedding_size: 2
          filters: # equal number of cnn layers
          - 2
          filter_size: # equal number of cnn layers
          - [1, 1]
          filter_stride: # equal number of cnn layers
          - [1, 1]
          pool_size: # equal number of cnn layers
          - [8, 8]
          tdnn_contexts:
          - 3
          - 3
          tdnn_dims:
          - 128
          - 128
          num_filters: 2
          linear_num: 2 # hidden number of linear layer
          cell_num: 2 # cell units of the lstm
          hidden1: 2 # number of hidden units of fully connected layer
          attention: false # whether to use attention, false mean use max-pooling
          attention_size: 64 # attention_size
          use_lstm_layer: false # whether to use lstm layer, false mean no lstm layer
          use_dropout: true # whether to use bn, dropout layer
          dropout_rate: 0.2
          use_bn: true # whether to use bn, dropout layer

          score_threshold: 0.5 # threshold to predict POS example
          threshold: 3 # threshold to predict POS example

    solver:
      name: SpeakerSolver
      quantization:
        enable: false # whether to quantization model
        quant_delay: 0 # Number of steps after which weights and activations are quantized during training
      adversarial:
        enable: false # whether to using adversiral training
        adv_alpha: 0.5 # adviseral alpha of loss
        adv_epslion: 0.1 # adviseral example epslion
      model_average:
        enable: false # use average model
        var_avg_decay: 0.99 # the decay rate of varaibles
      optimizer:
        name: adam
        epochs: 5 # maximum epochs
        batch_size: 4 # number of elements in a training batch
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
      postproc:
          name: SpeakerPostProc
          log_verbose: false
          eval: true # compute metrics
          infer: true  # get predict results
          pred_path: null # None for `model_path`/infer, dumps infer output to this dir
          thresholds:
              - 0.5
          smoothing:
              enable: true
              count: 2
      saver:
        model_path: "ckpt/emotion-speech-cls/test"
        max_to_keep: 10
        save_checkpoints_steps: 10
        keep_checkpoint_every_n_hours: 10000
        checkpoint_every: 10 # the step to save checkpoint
        summary: false
        save_summary_steps: 5
        eval_on_dev_every_secs: 1
        print_every: 10
        resume_model_path: ""
      run_config:
        debug: false # use tfdbug
        tf_random_seed: null # 0-2**32; null is None, try to read data from /dev/urandom if available or seed from the clock otherwise
        allow_soft_placement: true
        log_device_placement: false
        intra_op_parallelism_threads: 1
        inter_op_parallelism_threads: 1
        allow_growth: true
        log_step_count_steps: 1 #The frequency, in number of global steps, that the global step/sec and the loss will be logged during training.
      distilling:
        enable: false
        name : Teacher
        loss : DistillationLoss
        temperature: 5
        alpha: 0.5
        teacher_model: ''

    serving:
      enable: true
      name : Evaluate
      model: '' # saved model dir, ckpt dir, or frozen_model.pb
      inputs: 'inputs:0'
      outpus: 'softmax_output:0'
    '''

    # write config to file
    tempdir = self.get_temp_dir()
    #tempdir = 'bar'
    os.makedirs(tempdir, exist_ok=True)

    config_path = str(Path(tempdir).joinpath('speaker_task.yaml'))
    logging.info("config path: {}".format(config_path))
    with open(config_path, 'w', encoding='utf-8') as f:  #pylint: disable=invalid-name
      f.write(self.conf_str)

    # load config
    config = utils.load_config(config_path)
    logging.info("config: {}".format(config))

    # edit path in config
    dataset_path = Path(tempdir).joinpath('data')
    if not dataset_path.exists():
      dataset_path.mkdir()
    dataset_path_str = str(dataset_path)
    config['data']['train']['paths'] = [dataset_path_str]
    config['data']['eval']['paths'] = [dataset_path_str]
    config['data']['infer']['paths'] = [dataset_path_str]

    # generate dummy data
    feat_dim = config['data']['task']['audio']['feature_size']
    kaldi_dir_utils.gen_dummy_data_dir(
        dataset_path_str, 2, 2, feat_dim=feat_dim)

    solver_name = config['solver']['name']
    self.solver = registers.solver[solver_name](config)

    # config after process
    self.config = self.solver.config

  def tearDown(self):
    ''' tear down'''

  def test_generate_feat(self):
    ''' test generate feature'''
    task_name = self.config['data']['task']['name']
    task_class = registers.task[task_name]

    paths = []
    for mode in [utils.TRAIN, utils.EVAL, utils.INFER]:
      paths += self.config['data'][mode]['paths']

    task = task_class(self.config, utils.INFER)
    task.generate_feat(paths, dry_run=False)

  def test_generate_cmvn(self):
    ''' test generate cmvn'''
    tmp = tempfile.mktemp(suffix='cmvn.npy')
    self.config['data']['task']['audio']['cmvn_path'] = tmp

    self.config['data']['task']['suffix'] = '.wav'
    self.config['data']['task']['stride'] = 1.0
    paths = self.config['data'][utils.TRAIN]['paths']
    self.config['data'][utils.INFER]['paths'] = paths

    task_name = self.config['data']['task']['name']
    task_class = registers.task[task_name]

    task = task_class(self.config, utils.INFER)
    task.generate_cmvn(dry_run=False)

    self.assertTrue(
        os.path.exists(self.config['data']['task']['audio']['cmvn_path']))
    cmvn = np.load(self.config['data']['task']['audio']['cmvn_path'])
    self.assertEqual(cmvn.ndim, 4)

  def test_generate_data(self):
    ''' test generate data'''
    self.config['data']['task']['suffix'] = '.npy'

    task_name = self.config['data']['task']['name']
    task_class = registers.task[task_name]

    task = task_class(self.config, utils.TRAIN)

    for inputs, label, filename, clip_id, soft_labels in task.generate_data():
      logging.info(
          "feat shape:{} \n labels:{} \nfilename:{} \nclip_id:{}\nsoft_labels:{}"
          .format(inputs.shape, label, filename, clip_id, soft_labels))
      break

  #pylint: disable=too-many-locals
  def test_dataset(self):
    ''' dataset unittest'''
    batch_size = 4
    self.config['solver']['optimizer']['batch_size'] = batch_size

    task_name = self.config['data']['task']['name']
    task_class = registers.task[task_name]
    task = task_class(self.config, utils.TRAIN)

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

  def test_speaker_utt_task_dataset(self):
    task_name = 'SpeakerUttTask'
    self.config['data']['task']['name'] = task_name
    batch_size = 4
    self.config['solver']['optimizer']['batch_size'] = batch_size

    task_name = self.config['data']['task']['name']
    task_class = registers.task[task_name]

    for mode in (utils.TRAIN, utils.EVAL, utils.INFER):
      task = task_class(self.config, mode)
      dataset = task.input_fn(mode, batch_size, 1)()
      features, one_hot_labels = dataset.make_one_shot_iterator().get_next()
      samples = features['inputs']
      filenames = features['filepath']
      clip_ids = features['clipid']
      labels = features['labels']

      with self.cached_session(use_gpu=False, force_gpu=False) as sess:
        while True:
          batch_inputs, batch_labels, batch_files, batch_clipids, labels_onehot = \
             sess.run([samples, labels, filenames, clip_ids, one_hot_labels])

          del labels_onehot
          logging.info("feat shape: {}".format(batch_inputs.shape))
          logging.info("filename: {}".format(batch_files))
          logging.info("clip id: {}".format(batch_clipids))
          logging.info("labels: {}".format(batch_labels))
          break

  def test_speaker_utt_task_getitem(self):
    task_name = 'SpeakerUttTask'
    self.config['data']['task']['name'] = task_name
    task_class = registers.task[task_name]

    for mode in (utils.TRAIN, utils.EVAL, utils.INFER):
      task = task_class(self.config, mode)
      for i, (feats, labels) in enumerate(task):
        logging.info(f"SpkUttTask: __getitem__: {feats.keys()} {labels}")

  def test_speaker_utt_task_generate_data(self):
    task_name = 'SpeakerUttTask'
    self.config['data']['task']['name'] = task_name
    task_class = registers.task[task_name]
    for mode in (utils.TRAIN, utils.EVAL, utils.INFER):
      task = task_class(self.config, mode)
      logging.info(f"mode: {mode}")
      for i in range(10):
        for utt, segid, feat, spkid in task.generate_data():
          logging.info(
              f"SpkUttTask: generate_data: {utt} {segid} {feat.shape} {spkid}")


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.test.main()
