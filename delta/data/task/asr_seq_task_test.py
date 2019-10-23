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
''' Test for asr_seq_task.py '''

from pathlib import Path
from absl import logging
import numpy as np
import delta.compat as tf

from delta import utils
from delta.data.utils.test_utils import generate_json_data
from delta.utils.register import registers
from delta.utils.register import import_all_modules_for_register

# pylint: disable=missing-docstring


class AsrSeqTaskTest(tf.test.TestCase):
  ''' Unit test for AsrSeqTask. '''

  def setUp(self):
    super().setUp()
    self.conf_str = '''
    data:
      train:
        paths: null
        segments: null
      eval:
        paths: null
        segments: null
      infer:
        paths: null
        segments: null
      task:
        dummy: true # dummy inputs 
        name: AsrSeqTask
        type: asr # asr, tts
        audio:
          dry_run: false # not save feat
        src:
          max_len: 3000 # max length for frames
          subsampling_factor: 1
          preprocess_conf: null
        tgt:
          max_len: 100 # max length for target tokens
        vocab:
          type: char # char, bpe, wpm, word
          size: 3653 # vocab size in vocab_file
          path: '/nfs/cold_project/dataset/opensource/librispeech/espnet/egs/hkust/asr1/data/lang_1char/train_nodup_sp_units.txt' # path to vocab(default: 'vocab
        batch:
          batch_size: 32 # number of elements in a training batch
          batch_bins: 0 # maximum number of bins (frames x dim) in a trainin batch
          batch_frames_in: 0 # maximum number of input frames in a training batch
          batch_frames_out: 0 # maximum number of output frames in a training batch
          batch_frames_inout: 0 # maximum number of input+output frames in a training batch
          batch_strategy: auto # strategy to count maximum size of batch(support 4 values: "auto", "seq", "frame", "bin")
        batch_mode: false # ture, user control batch; false, `generate` will yeild one example 
        num_parallel_calls: 12
        num_prefetch_batch: 2
        shuffle_buffer_size: 200000
        need_shuffle: true
        sortagrad: true
        batch_sort_key: 'input' # shuffle, input, output for asr and tts, and sortagrad for asr
        num_batches: 0 # for debugging

    model:
      name: CTCAsrModel
      type: keras # raw, keras or eager model
      net:
        structure:
          encoder:
            name:
            filters: # equal number of cnn layers
            - 128
            - 512
            - 512
            filter_size: # equal number of cnn layers
            - [5, 3]
            - [5, 3]
            - [5, 3]
            filter_stride: # equal number of cnn layers
            - [1, 1]
            - [1, 1]
            - [1, 1]
            pool_size: # equal number of cnn layers
            - [4, 4]
            - [1, 2]
            - [1, 2]
            num_filters: 128
            linear_num: 786 # hidden number of linear layer
            cell_num: 128 # cell units of the lstm
            hidden1: 64 # number of hidden units of fully connected layer
            attention: false # whether to use attention, false mean use max-pooling
            attention_size: 128 # attention_size
            use_lstm_layer: false # whether to use lstm layer, false mean no lstm layer
            use_dropout: true # whether to use bn, dropout layer
            dropout_rate: 0.2
            use_bn: true # whether to use bn, dropout layer
          decoder:
            name: 
          attention:
            name:
    solver:
      name: AsrSolver
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
      distilling:
        enable: false 
        name : Teacher
        loss : DistillationLoss
        temperature: 5
        alpha: 0.5
        teacher_model: null # fronzen_graph.pb 
      optimizer:
        name: adam
        epochs: 5 # maximum epochs
        loss: CTCLoss 
        label_smoothing: 0.0 # label smoothing rate
        learning_rate:
          rate: 0.0001 # learning rate of Adam optimizer
          type:  exp_decay # learning rate type
          decay_rate: 0.99  # the lr decay rate
          decay_steps: 100  # the lr decay_step for optimizer
        clip_global_norm: 3.0 # clip global norm
        multitask: False # whether is multi-task
        early_stopping: # keras early stopping
          enable: true
          monitor: val_loss
          min_delta: 0
          patience: 5
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
          enbale: false
          name: EmoPostProc
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
        model_path: "ckpt/asr-seq/test"
        max_to_keep: 10
        save_checkpoints_steps: 100
        keep_checkpoint_every_n_hours: 10000
        checkpoint_every: 100 # the step to save checkpoint
        summary: false
        save_summary_steps: 100
        eval_on_dev_every_secs: 1
        print_every: 10
        resume_model_path: ""
      run_config:
        debug: false # use tfdbug
        tf_random_seed: null # 0-2**32; null is None, try to read data from /dev/urandom if available or seed from the clock otherwise
        allow_soft_placement: true
        log_device_placement: false
        intra_op_parallelism_threads: 10
        inter_op_parallelism_threads: 10
        allow_growth: true
        log_step_count_steps: 100 #The frequency, in number of global steps, that the global step/sec and the loss will be logged during training.
      run_options:
        trace_level: 3 # 0: no trace, 1: sotware trace, 2: hardware_trace, 3: full trace
        inter_op_thread_pool: -1
        report_tensor_allocations_upon_oom: true
    
    serving:
      enable: false 
      name : Evaluate
      model: null # saved model dir, ckpt dir, or frozen_model.pb
      inputs: 'inputs:0'
      outpus: 'softmax_output:0'   
    '''
    import_all_modules_for_register()
    tempdir = self.get_temp_dir()

    config_path = str(Path(tempdir).joinpath("asr_seq.yaml"))
    logging.info("config path: {}".format(config_path))
    with open(config_path, 'w', encoding='utf-8') as f:  #pylint: disable=invalid-name
      f.write(self.conf_str)

    self.config = utils.load_config(config_path)
    self.mode = utils.TRAIN
    self.batch_size = 4
    self.config['solver']['optimizer']['batch_size'] = self.batch_size

    #generate dummpy data
    nexamples = 10
    generate_json_data(self.config, self.mode, nexamples)

  def test_generate_data(self):
    for batch_mode in [True, False]:
      task_name = self.config['data']['task']['name']
      self.config['data']['task']['batch_mode'] = batch_mode
      self.config['data']['task']['dummy'] = False
      task = registers.task[task_name](self.config, self.mode)

      with self.cached_session(use_gpu=False, force_gpu=False):
        for uttid, feats, src_lens, targets, tgt_lens in task.generate_data():
          logging.debug('uttid : {}'.format(uttid))
          logging.debug("feats : {}, shape : {}".format(feats, feats.shape))
          logging.debug("targets : {}, shape : {}".format(
              targets, targets.shape))
          logging.debug('src_len : {}'.format(src_lens))
          logging.debug('tgt_len : {}'.format(tgt_lens))
          self.assertDTypeEqual(feats, np.float32)
          self.assertDTypeEqual(src_lens, np.int64)
          self.assertDTypeEqual(targets, np.int64)
          self.assertDTypeEqual(tgt_lens, np.int64)

          if batch_mode:
            self.assertEqual(len(uttid.shape), 1)
            self.assertEqual(len(feats.shape), 3)
            self.assertEqual(len(targets.shape), 2)
            self.assertEqual(len(src_lens.shape), 1)
            self.assertEqual(len(tgt_lens.shape), 1)
          else:
            self.assertEqual(tf.rank(uttid).numpy(), 0)
            self.assertEqual(len(feats.shape), 2)
            self.assertEqual(len(targets.shape), 1)
            self.assertEqual(tf.rank(src_lens).numpy(), 0)
            self.assertEqual(tf.rank(tgt_lens).numpy(), 0)

  def test_dataset(self):
    for batch_mode in [True, False]:
      task_name = self.config['data']['task']['name']
      self.config['data']['task']['batch_mode'] = batch_mode
      self.config['data']['task']['dummy'] = False
      task = registers.task[task_name](self.config, self.mode)

      with self.cached_session(use_gpu=False, force_gpu=False):
        for features, labels in task.dataset(
            self.mode, self.batch_size, epoch=1):  # pylint: disable=bad-continuation
          logging.debug("feats : {} : {}".format(features['inputs'],
                                                 features['inputs'].shape))
          logging.debug("ilens : {} : {}".format(
              features['input_length'], features['input_length'].shape))
          logging.debug("targets : {} : {}".format(features['targets'],
                                                   features['targets'].shape))
          logging.debug("olens : {} : {}".format(
              features['target_length'], features['target_length'].shape))
          logging.debug("ctc : {}, shape : {}".format(labels['ctc'],
                                                      labels['ctc'].shape))
          self.assertDTypeEqual(features['inputs'], np.float32)
          self.assertDTypeEqual(features['targets'], np.int32)
          self.assertDTypeEqual(features['input_length'], np.int32)
          self.assertDTypeEqual(features['target_length'], np.int32)

          self.assertEqual(len(features['inputs'].shape), 4)
          self.assertEqual(len(features['input_length'].shape), 1)
          self.assertEqual(len(features['targets'].shape), 2)
          self.assertEqual(len(features['target_length'].shape), 1)

  def test_dummy_dataset(self):
    for batch_mode in [True, False]:
      task_name = self.config['data']['task']['name']
      self.config['data']['task']['batch_mode'] = batch_mode
      self.config['data']['task']['dummy'] = True
      task = registers.task[task_name](self.config, self.mode)

      with self.cached_session(use_gpu=False, force_gpu=False):
        for _ in task.dataset(self.mode, self.batch_size, epoch=1):
          break
        for features, labels in task.dataset(
            self.mode, self.batch_size, epoch=1):  # pylint: disable=bad-continuation
          logging.debug("feats : {} : {}".format(features['inputs'],
                                                 features['inputs'].shape))
          logging.debug("ilens : {} : {}".format(
              features['input_length'], features['input_length'].shape))
          logging.debug("targets : {} : {}".format(features['targets'],
                                                   features['targets'].shape))
          logging.debug("olens : {} : {}".format(
              features['target_length'], features['target_length'].shape))
          logging.debug("ctc : {}, shape : {}".format(labels['ctc'],
                                                      labels['ctc'].shape))
          self.assertDTypeEqual(features['inputs'], np.float32)
          self.assertDTypeEqual(features['targets'], np.int32)
          self.assertDTypeEqual(features['input_length'], np.int32)
          self.assertDTypeEqual(features['target_length'], np.int32)

          self.assertEqual(len(features['inputs'].shape), 4)
          self.assertEqual(len(features['input_length'].shape), 1)
          self.assertEqual(len(features['targets'].shape), 2)
          self.assertEqual(len(features['target_length'].shape), 1)


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  tf.enable_eager_execution()
  tf.test.main()
