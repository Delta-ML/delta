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
''' espnet utils unittest'''
import tempfile
from pathlib import Path

import numpy as np
import delta.compat as tf
import h5py
import kaldiio
from absl import logging

# espnet
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.io_utils import SoundHDF5File
# test is package of espnet
from test.utils_test import make_dummy_json  #pylint: disable=wrong-import-order

# delta
from delta import utils
from delta.data.utils import espnet_utils
from delta.data.utils.test_utils import generate_json_data

#pylint: disable=invalid-name,too-many-locals,missing-docstring


class KaldiJsonReaderTest(tf.test.TestCase):

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

    tempdir = self.get_temp_dir()

    config_path = str(Path(tempdir).joinpath("asr_seq.yaml"))
    logging.info("config path: {}".format(config_path))
    with open(config_path, 'w', encoding='utf-8') as f:  #pylint: disable=invalid-name
      f.write(self.conf_str)

    self.config = utils.load_config(config_path)

  def tearDown(self):
    ''' tear down '''

  def test_make_batchset(self):
    dummy_json = make_dummy_json(128, [128, 512], [16, 128])
    for task in espnet_utils.TASK_SET:
      # check w/o adaptive batch size
      batchset = espnet_utils.make_batchset(
          task,
          data=dummy_json,
          batch_size=24,
          max_length_in=2**10,
          max_length_out=2**10,
          min_batch_size=1)
      self.assertEqual(
          sum([len(batch) >= 1 for batch in batchset]), len(batchset))
      logging.info('batch: {}'.format(([len(batch) for batch in batchset])))

      batchset = espnet_utils.make_batchset(
          task, dummy_json, 24, 2**10, 2**10, min_batch_size=10)
      self.assertEqual(
          sum([len(batch) >= 10 for batch in batchset]), len(batchset))
      logging.info('batch: {}'.format(([len(batch) for batch in batchset])))

      # check w/ adaptive batch size
      batchset = espnet_utils.make_batchset(
          task, dummy_json, 24, 256, 64, min_batch_size=10)
      self.assertEqual(
          sum([len(batch) >= 10 for batch in batchset]), len(batchset))
      logging.info('batch: {}'.format(([len(batch) for batch in batchset])))

      batchset = espnet_utils.make_batchset(
          task, dummy_json, 24, 256, 64, min_batch_size=10)
      self.assertEqual(
          sum([len(batch) >= 10 for batch in batchset]), len(batchset))

  def test_sortagrad(self):
    dummy_json = make_dummy_json(128, [1, 700], [1, 700])

    for task in espnet_utils.TASK_SET:
      if task == 'tts':
        batchset = espnet_utils.make_batchset(
            task,
            dummy_json,
            16,
            2**10,
            2**10,
            batch_sort_key="input",
            shortest_first=True)
        key = 'output'
      elif task == 'asr':
        batchset = espnet_utils.make_batchset(
            task,
            dummy_json,
            16,
            2**10,
            2**10,
            batch_sort_key='input',
            shortest_first=True)
        key = 'input'

      prev_start_ilen = batchset[0][0][1][key][0]['shape'][0]
      for batch in batchset:
        # short to long
        cur_start_ilen = batch[0][1][key][0]['shape'][0]
        self.assertGreaterEqual(cur_start_ilen, prev_start_ilen)

        prev_ilen = cur_start_ilen
        for sample in batch:
          cur_ilen = sample[1][key][0]['shape'][0]
          # long to short in minibatch
          self.assertLessEqual(cur_ilen, prev_ilen)
          prev_ilen = cur_ilen
        prev_start_ilen = cur_start_ilen

  def test_load_inputs_and_targets_legacy_format(self):
    # batch = [("F01_050C0101_PED_REAL",
    #          {"input": [{"feat": "some/path.ark:123"}],
    #           "output": [{"tokenid": "1 2 3 4"}],

    tmpdir = Path(tempfile.mkdtemp())
    ark = str(tmpdir.joinpath('test.ark'))
    scp = str(tmpdir.joinpath('test.scp'))

    desire_xs = []
    desire_ys = []
    with kaldiio.WriteHelper('ark,scp:{},{}'.format(ark, scp)) as f:
      for i in range(10):
        x = np.random.random((100, 100)).astype(np.float32)
        uttid = 'uttid{}'.format(i)
        f[uttid] = x
        desire_xs.append(x)
        desire_ys.append(np.array([1, 2, 3, 4]))

    batch = []
    with open(scp, 'r') as f:
      for line in f:
        uttid, path = line.strip().split()
        batch.append((uttid, {
            'input': [{
                'feat': path,
                'name': 'input1'
            }],
            'output': [{
                'tokenid': '1 2 3 4',
                'name': 'target1'
            }]
        }))

    logging.info("kaldi meta: {}".format(batch[0]))
    load_inputs_and_targets = LoadInputsAndTargets()
    xs, ys = load_inputs_and_targets(batch)
    for x, xd in zip(xs, desire_xs):
      self.assertAllEqual(x, xd)
    for y, yd in zip(ys, desire_ys):
      self.assertAllEqual(y, yd)

  def test_load_inputs_and_targets_new_format(self):
    # batch = [("F01_050C0101_PED_REAL",
    #           {"input": [{"feat": "some/path.h5",
    #                       "filetype": "hdf5"}],
    #           "output": [{"tokenid": "1 2 3 4"}],

    tmpdir = tempfile.mkdtemp()
    p = Path(tmpdir).joinpath('test.h5')

    desire_xs = []
    desire_ys = []
    batch = []
    with h5py.File(str(p), 'w') as f:
      # batch: List[Tuple[str, Dict[str, List[Dict[str, Any]]]]]
      for i in range(10):
        x = np.random.random((100, 100)).astype(np.float32)
        uttid = 'uttid{}'.format(i)
        f[uttid] = x
        batch.append((uttid, {
            'input': [{
                'feat': str(p) + ':' + uttid,
                'filetype': 'hdf5',
                'name': 'input1'
            }],
            'output': [{
                'tokenid': '1 2 3 4',
                'name': 'target1'
            }]
        }))
        desire_xs.append(x)
        desire_ys.append(np.array([1, 2, 3, 4]))

    logging.info("h5py meta: {}".format(batch[0]))
    load_inputs_and_targets = LoadInputsAndTargets()
    xs, ys = load_inputs_and_targets(batch)
    for x, xd in zip(xs, desire_xs):
      self.assertAllEqual(x, xd)
    for y, yd in zip(ys, desire_ys):
      self.assertAllEqual(y, yd)

  def test_sound_hdf5_file(self):
    tmpdir = Path(tempfile.mkdtemp())
    for fmt in ['flac', 'wav']:
      valid = {
          'a': np.random.randint(-100, 100, 25, dtype=np.int16),
          'b': np.random.randint(-1000, 1000, 100, dtype=np.int16)
      }

      # Note: Specify the file format by extension
      p = tmpdir.joinpath('test.{}.h5'.format(fmt))
      p.touch(exist_ok=True)
      p = str(p.resolve())
      f = SoundHDF5File(p, 'a')

      for k, v in valid.items():
        f[k] = (v, 8000)

      for k, v in valid.items():
        t, r = f[k]
        self.assertEqual(r, 8000)
        self.assertAllEqual(t, v)

  def test_converter(self):
    np.random.seed(100)

    nexamples_list = (10, 12)
    batch_size = 4
    self.config['solver']['optimizer']['batch_size'] = batch_size
    converter = espnet_utils.ASRConverter(self.config)

    for mode in (utils.TRAIN, utils.EVAL, utils.INFER):
      for nexamples in nexamples_list:
        desire_xs, desire_ilens, desire_ys, desire_olens = generate_json_data(
            self.config, mode, nexamples)
        del desire_xs, desire_ilens, desire_ys, desire_olens
        data_metas = espnet_utils.get_batches(self.config, mode)

        batches = data_metas['data']
        n_utts = data_metas['n_utts']
        self.assertEqual(n_utts, nexamples)

        o_uttids = []
        o_xs = []
        o_ilens = []
        o_ys = []
        o_olens = []
        for _, batch in enumerate(batches):
          batch_data = converter(batch)
          self.assertEqual(len(batch_data), 5)

          xs, ilens, ys, olens, uttids = batch_data
          for x, ilen, y, olen, uttid in zip(xs, ilens, ys, olens, uttids):
            self.assertDTypeEqual(x, np.float32)
            self.assertDTypeEqual(y, np.int64)
            o_uttids.append(uttid)
            o_xs.append(x)
            o_ilens.append(ilen)
            o_ys.append(y)
            o_olens.append(olen)

        self.assertEqual(len(o_xs), nexamples)


if __name__ == '__main__':
  tf.test.main()
