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
''' Speaker Classification. '''
import os
import random
import multiprocessing as mp
from pathlib import Path
from absl import logging
import numpy as np
import tensorflow as tf
import kaldiio

from delta import utils
from delta.utils.register import registers
from delta.data.task.base_speech_task import SpeechTask

#pylint: disable=too-many-instance-attributes
#pylint: disable=too-many-arguments
#pylint: disable=too-many-locals
#pylint: disable=too-few-public-methods
#pylint: disable=arguments-differ


class UttMeta():
  ''' An utterance. '''

  def __init__(self, feat_scp, label):
    self.feat_scp = feat_scp
    self.label = label


class SpeakerMetaData():
  ''' Meta data for speaker corpus. '''

  def __init__(self):
    self.utt2wav = {}
    self.utt2wavlen = {}
    self.utt2feat = {}
    self.utt2featlen = {}
    self.utt2vad = {}
    self.utt2spk = {}
    self.spk2utt = {}
    self.spk2numutt = {}
    self.spk2id = {}
    self.utts = {}

  def collect(self):
    ''' Collect various meta data. '''
    assert len(self.utt2feat) == \
           len(self.utt2featlen) == \
           len(self.utt2vad) == \
           len(self.utt2spk)
    for spk in sorted(self.spk2utt.keys()):
      self.spk2id[spk] = len(self.spk2id)
    all_utt_keys = list(self.utt2feat.keys())
    random.shuffle(all_utt_keys)
    for utt_key in all_utt_keys:
      spk = self.utt2spk[utt_key]
      utt = {
          'key': utt_key,
          'feat': self.utt2feat[utt_key],
          'featlen': self.utt2featlen[utt_key],
          'vad': self.utt2vad[utt_key],
          'spk': spk,
          'spkid': self.spk2id[spk]
      }
      self.utts[utt_key] = utt


class ChunkSampler():
  ''' Produce samples from utterance. '''

  def __init__(self, chunk_size):
    '''
      Args:
      chunk_size: fixed chunk size in frames.
      pad_chunks: whether to zero pad chunks.
    '''
    self.chunk_size = chunk_size
    self.pad_chunks = True

  def set_chunk_size(self, chunk_size):
    ''' Set chunk size. '''
    self.chunk_size = chunk_size

  def set_pad_chunks(self, pad_chunks):
    ''' Set whether to zero pad chunks. '''
    self.pad_chunks = pad_chunks

  def utt_to_samples(self, args):
    '''
    Utt to samples of (feat_chunk, spk_id, sample_key).
    Will be run in a process pool so restrictions apply.
    '''
    result_queue, utt_info = args
    logging.debug(utt_info)

    # TODO: wrap into a function or something
    utt_key, utt_meta = utt_info

    # Load features and select voiced frames.
    feat_scp = utt_meta['feat']
    feat_mat = kaldiio.load_mat(feat_scp)
    vad_scp = utt_meta['vad']
    vad_mat = kaldiio.load_mat(vad_scp)
    num_frames_feat, feat_dim = feat_mat.shape
    num_frames_vad = vad_mat.shape[0]
    logging.debug('feat_mat: %s, vad_mat: %s' %
                  (str(feat_mat.shape), str(vad_mat.shape)))
    if num_frames_feat != num_frames_vad:
      logging.debug('num_frames_feat != num_frames_vad: %d vs %d' %
                    (num_frames_feat, num_frames_vad))
      return None
    voiced_frames_index = np.where(vad_mat == 1)[0]
    logging.debug('voiced_frames_index: %s' % (str(voiced_frames_index.shape)))
    feat_mat_voiced = feat_mat[voiced_frames_index, :]
    num_frames = feat_mat_voiced.shape[0]
    logging.debug('feat_mat_voiced: %s' % (str(feat_mat_voiced.shape)))

    spk_id = utt_meta['spkid']

    logging.debug('Chunk size: %d' % (self.chunk_size))

    results = []
    chunk_idx = 0
    for offset in range(0, num_frames, self.chunk_size):
      logging.debug('offset = %d' % (offset))
      feat_chunk = feat_mat_voiced[offset:offset + self.chunk_size, :]
      unpadded_frames = feat_chunk.shape[0]
      if self.pad_chunks and unpadded_frames < self.chunk_size:
        logging.debug('Padding chunk of frames %d ...' % (unpadded_frames))
        padded = np.zeros((self.chunk_size, feat_dim), dtype=feat_chunk.dtype)
        padded[:unpadded_frames, :] = feat_chunk
        feat_chunk = padded
      feat_chunk = np.expand_dims(feat_chunk, axis=2)  # TODO: not here
      sample_key = '%s_chunk%02d' % (utt_key, chunk_idx)
      sample = (feat_chunk, spk_id, sample_key)
      chunk_idx += 1
      results.append(sample)
    if result_queue:
      # queue mode
      result_queue.put(results)
      return None
    # imap mode
    return results


class DataQueueAsync():
  ''' Sample from raw data. '''

  def __init__(self, meta, sampler, num_processes=None, max_qsize=20):
    '''
    Params:
      num_processes: number of IO processes. Default = None (# CPU cores).
      max_qsize: multiprocessing.Queue(maxsize=max_qsize).
      method: queue(use manager.Queue), async_result(AsyncResult.get())
    '''

  def start(self):
    ''' Start sampling async. '''

  def get_items(self):
    ''' Get a generator of results. '''


class ManagerQueueDataQueue(DataQueueAsync):
  '''
  A DataQueueAsync implementation using multiprocessing.Manager to
  establish a Queue between Pool workers and host process.
  Its efficiency is quite low compared to the other implementation.
  And speed is even lower than a local single thread implementation.
  THe bottleneck may be the Queue constantly locking and blocking.
  '''

  def __init__(self, meta, sampler, num_processes=None, max_qsize=20):
    super().__init__(meta, sampler, num_processes, max_qsize)
    self.meta = meta
    self.sampler = sampler
    self.pool = None
    self.pool_res = None
    self.num_processes = num_processes
    self.mp_manager = mp.Manager()
    self.mp_queue = self.mp_manager.Queue(max_qsize)

  def start(self):
    ''' Start sampling async. '''
    self.pool = mp.Pool(self.num_processes)
    pool_args = [(self.mp_queue, x) for x in self.meta.utts.items()]
    self.pool_res = self.pool.map_async(
        self.sampler.utt_to_samples, pool_args, chunksize=100)
    self.pool.close()

  def done(self):
    ''' Whether all data have been consumed. '''
    return self.pool_res.ready() and self.mp_queue.qsize() == 0

  def get_items(self):
    ''' Get a generator of results. '''
    while not self.done():
      yield self.mp_queue.get()


class ImapUnorderedDataQueue(DataQueueAsync):
  '''
  Use async imap() calls to fetch partial results.
  Much faster than the Manager-based one.
  '''

  def __init__(self, meta, sampler, num_processes=None, max_qsize=20):
    super().__init__(meta, sampler, num_processes, max_qsize)
    self.meta = meta
    self.meta_keys = []
    self.meta_keys_pointer = 0
    self.end_of_inputs = False
    self.sampler = sampler
    self.pool = None
    self.pool_chunk_size = 1000
    self.pool_res = None
    self.num_processes = num_processes

  def start(self):
    ''' Start sampling async. '''
    self.pool = mp.Pool(self.num_processes)
    self.meta_keys = list(self.meta.utts.keys())
    logging.info('Shuffling utt keys ...')
    random.shuffle(self.meta_keys)

  def next_chunk(self):
    ''' Return an iterator of next chunk of utterances. '''
    for index in range(self.meta_keys_pointer,
                       self.meta_keys_pointer + self.pool_chunk_size):
      if index < len(self.meta_keys):
        utt_key = self.meta_keys[index]
        utt_meta = self.meta.utts[utt_key]
        yield (utt_key, utt_meta)
      else:
        self.end_of_inputs = True
    self.meta_keys_pointer = index + 1

  def append_requests(self):
    ''' Append some requests to pool. '''
    pool_args = [(None, x) for x in self.next_chunk()]
    self.pool_res = self.pool.imap_unordered(
        self.sampler.utt_to_samples, pool_args, chunksize=100)

  def get_items(self):
    ''' Get a generator of results. '''
    while True:
      self.append_requests()
      for item in self.pool_res:
        yield item
      if self.end_of_inputs:
        break
    self.pool.close()


@registers.task.register
class SpeakerClsTask(SpeechTask):
  ''' Speaker Classification Task '''

  def __init__(self, config, mode):
    super().__init__(config, mode)

    self.mode = mode
    self.dataconf = self.config['data']
    self.taskconf = self.dataconf['task']
    self.solverconf = self.config['solver']

    self.data_type = self.taskconf['data_type']

    self.chunk_size_seconds = self.taskconf['audio']['clip_size']
    # TODO: configurable frame rate
    self.chunk_size_frames = self.chunk_size_seconds * 100
    self.feature_dims = self.taskconf['audio']['feature_size']
    # TODO: delta features
    self.feature_shape = (self.chunk_size_frames, self.feature_dims, 1)
    # TODO: text input is useless
    self.max_text_len = 10

    self._cmvn_path = self.taskconf['audio']['cmvn_path']

    # meta data
    self.meta = SpeakerMetaData()
    self._classes = {}

    logging.info('Loading meta data ...')
    self.load_meta_data()

    self.sampler = ChunkSampler(self.chunk_size_frames)

  def load_meta_data(self):
    ''' Load meta data. '''
    data_paths = self.dataconf[self.mode]['paths']
    logging.info('Loading mode %s dirs: %s ...' % (self.mode, data_paths))
    for data_path in data_paths:
      logging.info('Loading dir %s ...' % (data_path))
      if self.data_type == 'KaldiDataDirectory':
        self.load_kaldi_data_directory(data_path)
      else:
        raise ValueError('Unsupported data type: %s' % (self.data_type))
    logging.info('Loaded mode %s: %d features, %d vads, %d utt2spk, %d spks.' %
                 (self.mode, len(self.meta.utt2feat), len(self.meta.utt2vad),
                  len(self.meta.utt2spk), len(self.meta.spk2utt)))
    self.meta.collect()
    self._classes = self.meta.spk2id

  def load_kaldi_data_directory(self, data_path):
    ''' Load meta data from Kaldi data directory. '''

    def scp_to_dict(file_path, target_dict):
      ''' Parse a scp file and target_dict[key] = value. '''
      with open(file_path, 'r') as fp_in:
        for line in fp_in:
          tokens = line.strip().split()
          if len(tokens) != 2:
            continue
          target_dict[tokens[0]] = tokens[1]

    def utt2spk_to_spk2utt(utt2spk, spk2utt, spk2numutt):
      ''' Convert utt2spk to spk2utt. '''
      for utt, spk in utt2spk.items():
        if spk not in spk2utt:
          spk2utt[spk] = []
        spk2utt[spk].append(utt)
        if spk not in spk2numutt:
          spk2numutt[spk] = 0
        spk2numutt[spk] += 1

    scp_to_dict(os.path.join(data_path, 'feats.scp'), self.meta.utt2feat)
    scp_to_dict(os.path.join(data_path, 'feats.len'), self.meta.utt2featlen)
    scp_to_dict(os.path.join(data_path, 'vad.scp'), self.meta.utt2vad)
    scp_to_dict(os.path.join(data_path, 'utt2spk'), self.meta.utt2spk)
    utt2spk_to_spk2utt(self.meta.utt2spk, self.meta.spk2utt,
                       self.meta.spk2numutt)
    logging.info('Loaded dir %s: %d features, %d vads, %d utt2spk, %d spks.' %
                 (data_path, len(self.meta.utt2feat), len(self.meta.utt2vad),
                  len(self.meta.utt2spk), len(self.meta.spk2utt)))

  def test_load_funcs(self):
    ''' Test kaldiio's functions. '''

    def load_func(ark_offset):
      return kaldiio.load_mat(ark_offset)

    for utt in self.meta.utt2feat:
      ark_offset = self.meta.utt2feat[utt]
      feat_arr = load_func(ark_offset)
      logging.info('feat_arr: %s' % (str(feat_arr.shape)))

  @property
  def num_class(self):
    ''' Return number of classes. '''
    return len(self._classes)

  @property
  def classes(self):
    ''' Return a map from class names to label ids. '''
    return self._classes

  def class_id(self, class_name):
    ''' Return the numeric label of a given class name. '''
    return self._classes[class_name]

  def generate_feat(self, filelist, dry_run=False):
    ''' Stub. Not implemented because we use Kaldi features. '''

  def generate_cmvn(self, filelist=None, dry_run=False):
    ''' Generate mean and vars of features. '''
    sums, square, count = utils.create_cmvn_statis(
        self.taskconf['audio']['feature_size'],
        self.taskconf['audio']['add_delta_deltas'])

    self.sampler.set_chunk_size(100000)
    self.sampler.set_pad_chunks(False)
    for inputs, _, _, _, _, _ in \
        self.generate_data():
      # update stats
      if inputs.ndim == 3:
        inputs = np.expand_dims(inputs, axis=0)
      sums, square, count = utils.update_cmvn_statis(
          inputs, sums, square, count, axis=(0, 1))
    # compute cmvn
    mean, var = utils.compute_cmvn(sums, square, count)
    if dry_run:
      logging.info('save cmvn:{}'.format(self._cmvn_path))
    else:
      np.save(self._cmvn_path, (mean, var))
    logging.info('generate cmvn done')
    logging.info(mean)
    logging.info(var)

  def generate_data(self):
    '''
    Yields samples.

    Args:
      multiprocess: use multiprocessing. Default = True.

    Yields:
      (inputs, texts, label, filename, clip_id, soft_label)
    '''
    class_num = self.taskconf['classes']['num']

    def process_sample(sample):
      inputs, label, utt_key = sample
      texts = np.array([0] * self.max_text_len)
      filename = utt_key
      clip_id = 0
      soft_label = np.zeros((1,))  # disabled for speaker model
      return inputs, texts, label, filename, clip_id, soft_label

    if self.mode == utils.INFER:
      # Estimator.predict might cause multiprocessing to fail.
      multiprocess = False
    else:
      multiprocess = True

    if multiprocess:
      q = ImapUnorderedDataQueue
      data_queue = q(self.meta, self.sampler, num_processes=4)
      data_queue.start()
      for samples in data_queue.get_items():
        for sample in samples:
          yield process_sample(sample)
    else:
      for item in self.meta.utts.items():
        samples = self.sampler.utt_to_samples((None, item))
        for sample in samples:
          yield process_sample(sample)
    raise StopIteration

  def feature_spec(self):
    class_num = self.taskconf['classes']['num']
    output_shapes = (
        tf.TensorShape(self.feature_shape),  # audio_feat e.g. (3000, 40, 3)
        tf.TensorShape([self.max_text_len]),  # text
        tf.TensorShape([]),  # label
        tf.TensorShape([]),  # filename
        tf.TensorShape([]),  # clip_id
        tf.TensorShape([1]),  # soft_label, disabled for speaker model
    )
    output_types = (
        tf.float32,
        tf.int32,
        tf.int32,
        tf.string,
        tf.int32,
        tf.float32,
    )
    return output_shapes, output_types

  def preprocess_batch(self, batch):
    return batch

  def dataset(self, mode, batch_size, num_epoch):
    shapes, types = self.feature_spec()
    data = tf.data.Dataset.from_generator(
        generator=lambda: self.generate_data(),
        output_types=types,
        output_shapes=shapes,
    )

    if 'shuffle_buffer_size' in self.solverconf['optimizer']:
      buffer_size = self.solverconf['optimizer']['shuffle_buffer_size']
    else:
      buffer_size = 1000
    logging.info(
        'Using buffer size of %d batches = %d in shuffle_and_repeat().' %
        (buffer_size, buffer_size * batch_size))
    if mode == utils.TRAIN:
      data = data.apply(
          tf.data.experimental.shuffle_and_repeat(
              buffer_size=buffer_size * batch_size, count=1, seed=None))

    def make_example(inputs, texts, labels, filenames, clip_ids, soft_labels):
      features = {
          'inputs': inputs,
          'texts': texts,
          'labels': labels,
          'filepath': filenames,
          'clipid': clip_ids,
          'soft_labels': soft_labels,
      }
      return features, labels

    return data.apply(
        tf.data.experimental.map_and_batch(
            make_example, batch_size,
            drop_remainder=False)).prefetch(tf.contrib.data.AUTOTUNE)


if __name__ == '__main__':

  def test_run():
    os.environ['CUDA_VISIBILE_DEVICES'] = ''
    logging.set_verbosity(logging.DEBUG)

    main_root = os.environ['MAIN_ROOT']
    main_root = Path(main_root)
    config_file = main_root.joinpath('speaker-test.yml')
    config = utils.load_config(config_file)

    solver_name = config['solver']['name']
    solver = registers.solver[solver_name](config)

    # config after process
    config = solver.config

    test_samples = False
    if test_samples:
      config['data']['task']['suffix'] = '.wav'
      data = SpeakerClsTask(config, utils.TRAIN)
    else:
      config['data']['task']['suffix'] = '.npy'
      data = SpeakerClsTask(config, utils.TRAIN)

    num_samples = 0
    log_samples = False
    for inputs, texts, label, filename, clip_id, soft_labels in \
        data.generate_data():
      if log_samples:
        logging.info(
            "feat shape:{} \ntext: {} \nlabels:{} \nfilename:{} \nclip_id:{}\nsoft_labels:{}"
            .format(inputs.shape, texts, label, filename, clip_id, soft_labels))
      if num_samples % 100 == 0:
        logging.info('Processed %d samples.' % (num_samples))
      num_samples += 1

    logging.info('Processed %d samples.' % (num_samples))

    batch_size = 4
    config['solver']['optimizer']['batch_size'] = batch_size
    dataset = data.input_fn(utils.TRAIN, batch_size, 1)()
    features, labels = dataset.make_one_shot_iterator().get_next()
    samples = features['inputs']
    filenames = features['filepath']
    clip_ids = features['clipid']
    soft_labels = features['soft_labels']

    num_samples = 0
    log_samples = False
    with tf.Session() as sess:
      try:
        while True:
          batch_inputs, batch_labels, batch_files, batch_clipids, \
            labels_onehot, batch_soft_labels = \
            sess.run([samples, labels, filenames, clip_ids, tf.one_hot(labels, 2), soft_labels])
          if log_samples:
            logging.debug("feat shape: {}".format(batch_inputs.shape))
            logging.debug("labels: {}".format(batch_labels))
            logging.debug("filename: {}".format(batch_files))
            logging.debug("clip id: {}".format(batch_clipids))
            logging.debug("onehot: {}".format(labels_onehot))
            logging.debug("soft_labels: {}".format(batch_soft_labels))
          if num_samples % 100 == 0:
            logging.info('Processed %d samples.' % (num_samples))
          num_samples += 1
      except tf.errors.OutOfRangeError as ex:
        logging.info(ex)
    logging.info('Processed %d samples.' % (num_samples))

  test_run()
