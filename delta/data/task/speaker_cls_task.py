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
import random
import multiprocessing as mp
from absl import logging
import numpy as np
import delta.compat as tf
import kaldiio
from collections import defaultdict
from pathlib import Path

from delta import utils
from delta.utils.register import registers
from delta.utils.kaldi.kaldi_dir import KaldiMetaData
from delta.data.task.base_speech_task import SpeechTask

#pylint: disable=too-many-instance-attributes
#pylint: disable=too-many-arguments
#pylint: disable=too-many-locals
#pylint: disable=too-few-public-methods
#pylint: disable=arguments-differ


class ChunkSampler():
  ''' Produce samples from utterance. '''

  def __init__(self, meta, chunk_size):
    '''
      Args:
      meta: an object of `KaldiMetaData` or None
      chunk_size: fixed chunk size in frames.
    '''
    self.chunk_size = chunk_size
    self.pad_chunks = True
    self.add_random_offset = False
    self.drop_short_chunks = 0.0
    self.single_chunk = False

    self.meta = meta
    self.spk_keys = None
    self.spk_keys_pointer = 0
    self.utt_keys = None
    self.meta_keys_pointer = 0
    self.select_by_spk = True
    self.num_repeats = 1

  def copy(self):
    ''' Return a copy of self, except meta data. '''
    new = ChunkSampler(None, self.chunk_size)
    new.pad_chunks = self.pad_chunks
    new.add_random_offset = self.add_random_offset
    new.drop_short_chunks = self.drop_short_chunks
    new.single_chunk = self.single_chunk
    new.select_by_spk = self.select_by_spk
    return new

  def get_bare_sampler(self):
    ''' Return a new sampler with no meta data. '''
    new = self.copy()
    return new.utt_to_samples

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
    num_frames_feat, feat_dim = feat_mat.shape
    vad_scp = utt_meta['vad']

    if vad_scp:
      vad_mat = kaldiio.load_mat(vad_scp)
      num_frames_vad = vad_mat.shape[0]
      logging.debug('feat_mat: %s, vad_mat: %s' %
                    (str(feat_mat.shape), str(vad_mat.shape)))
      if num_frames_feat != num_frames_vad:
        logging.debug('num_frames_feat != num_frames_vad: %d vs %d' %
                      (num_frames_feat, num_frames_vad))
        return None
      voiced_frames_index = np.where(vad_mat == 1)[0]
      logging.debug('voiced_frames_index: %s' %
                    (str(voiced_frames_index.shape)))
      feat_mat_voiced = feat_mat[voiced_frames_index, :]
    else:
      # If no VAD info was found, the entire utt will be used.
      feat_mat_voiced = feat_mat
    num_frames_voiced = feat_mat_voiced.shape[0]
    logging.debug('feat_mat_voiced: %s' % (str(feat_mat_voiced.shape)))

    spk_id = utt_meta['spkid']

    logging.debug('Chunk size: %d' % (self.chunk_size))

    results = []
    chunk_idx = 0
    if self.add_random_offset:
      random_offset = np.random.randint(0, self.chunk_size)
    else:
      random_offset = 0
    for offset in range(random_offset, num_frames_voiced, self.chunk_size):
      if self.single_chunk:
        available = num_frames_voiced - self.chunk_size
        if available < 0:
          # No padding.
          logging.warn('Single chunk mode: available < 0.')
          break
        offset = random.randint(0, available)
      logging.debug('offset = %d' % (offset))
      feat_chunk = feat_mat_voiced[offset:offset + self.chunk_size, :]
      unpadded_frames = feat_chunk.shape[0]
      if self.pad_chunks and unpadded_frames < self.chunk_size:
        rel_chunk_len = float(unpadded_frames) / self.chunk_size
        if rel_chunk_len < self.drop_short_chunks:
          continue
        logging.debug('Padding chunk of frames %d ...' % (unpadded_frames))
        padded = np.zeros((self.chunk_size, feat_dim), dtype=feat_chunk.dtype)
        padded[:unpadded_frames, :] = feat_chunk
        feat_chunk = padded
      feat_chunk = np.expand_dims(feat_chunk, axis=2)  # TODO: not here
      sample_key = '%s_chunk%02d' % (utt_key, chunk_idx)
      sample = (feat_chunk, spk_id, sample_key)
      chunk_idx += 1
      results.append(sample)
      if self.single_chunk:
        break
    if result_queue:
      # queue mode
      result_queue.put(results)
      return None
    # imap mode
    return results

  def reset(self):
    ''' Reset progress. '''
    if self.select_by_spk:
      logging.info('Shuffling spk keys ...')
      self.spk_keys = []
      for index in range(self.num_repeats):
        temp_keys = list(self.meta.spks.keys())
        random.shuffle(temp_keys)
        self.spk_keys.extend(temp_keys)
    else:
      logging.info('Shuffling utt keys ...')
      self.utt_keys = []
      for index in range(self.num_repeats):
        temp_keys = list(self.meta.utts.keys())
        random.shuffle(temp_keys)
        self.utt_keys.extend(temp_keys)

  def next_part_utts(self, part_size):
    ''' Return next part of utts. '''
    if self.select_by_spk:
      # Select utterances by speaker.
      for index in range(part_size):
        try:
          spk = self.spk_keys.pop()
        except IndexError:
          break
        spk_utts = self.meta.spks[spk].utts
        if not spk_utts:
          # None (e.g. with external id mapping) or len == 0
          continue
        random_idx = random.randint(0, len(spk_utts) - 1)
        if random_idx < 0:
          # Not likely.
          continue
        utt_key = spk_utts[random_idx]
        utt_meta = self.meta.utts[utt_key]
        yield (utt_key, utt_meta)
    else:
      # Randomly selects utterances.
      for index in range(part_size):
        try:
          utt_key = self.utt_keys.pop()
        except IndexError:
          break
        utt_meta = self.meta.utts[utt_key]
        yield (utt_key, utt_meta)


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
        self.sampler.get_bare_sampler(), pool_args, chunksize=100)
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
    self.meta = meta  # TODO: remove
    self.sampler = sampler
    self.pool = None
    self.pool_chunk_size = 1000
    self.pool_res = None
    self.num_processes = num_processes

  def start(self):
    ''' Start sampling async. '''
    self.pool = mp.Pool(self.num_processes)

  def append_requests(self):
    '''
    Append some requests to pool.

    Returns:
      a bool, True if still has unconsumed data, False otherwise.
    '''
    pool_args = [
        (None, x) for x in self.sampler.next_part_utts(self.pool_chunk_size)
    ]
    self.pool_res = self.pool.imap_unordered(
        self.sampler.get_bare_sampler(), pool_args, chunksize=100)
    if pool_args:
      return True
    else:
      return False

  def get_items(self):
    ''' Get a generator of results. '''
    while True:
      has_remaining = self.append_requests()
      for item in self.pool_res:
        yield item
      if not has_remaining:
        break
    self.pool.close()


@registers.task.register
class SpeakerClsTask(SpeechTask):
  ''' Speaker Classification Task '''

  def __init__(self, config, mode):
    super().__init__(config, mode)
    self.dataconf = self.config['data']
    self.taskconf = self.dataconf['task']
    self.solverconf = self.config['solver']

    self.data_type = self.taskconf['data_type']

    if 'whole_utt_inference' in self.taskconf['audio']:
      self.whole_utt_inference = self.taskconf['audio']['whole_utt_inference']
    else:
      self.whole_utt_inference = False

    if 'add_random_offset' in self.taskconf['audio']:
      self.add_random_offset = self.taskconf['audio']['add_random_offset']
    else:
      self.add_random_offset = False

    if 'drop_short_chunks' in self.taskconf['audio']:
      self.drop_short_chunks = self.taskconf['audio']['drop_short_chunks']
    else:
      self.drop_short_chunks = 0.0

    if 'single_chunk' in self.taskconf['audio']:
      self.single_chunk = self.taskconf['audio']['single_chunk']
    else:
      self.single_chunk = 0.0

    if 'select_by_spk_train' in self.taskconf['audio']:
      self.select_by_spk_train = self.taskconf['audio']['select_by_spk_train']
    else:
      self.select_by_spk_train = False

    if 'select_by_spk_eval' in self.taskconf['audio']:
      self.select_by_spk_eval = self.taskconf['audio']['select_by_spk_eval']
    else:
      self.select_by_spk_eval = False

    if 'num_repeats' in self.taskconf['audio']:
      self.num_repeats = self.taskconf['audio']['num_repeats']
    else:
      self.num_repeats = 1

    self.chunk_size_seconds = self.taskconf['audio']['clip_size']
    # TODO: configurable frame rate
    self.chunk_size_frames = self.chunk_size_seconds * 100
    self.feature_dims = self.taskconf['audio']['feature_size']
    # TODO: delta features
    if self.mode == utils.INFER and self.whole_utt_inference:
      self.feature_shape = (None, self.feature_dims, 1)
    else:
      self.feature_shape = (self.chunk_size_frames, self.feature_dims, 1)
    self.feature_shape = (None, self.feature_dims, 1)

    # TODO: not implemented
    self.uniform_resample = False

    # 10k sample/utts is somehow enough.
    self.cmvn_max_samples = 10000

    self._cmvn_path = self.taskconf['audio']['cmvn_path']

    # meta data
    self.meta = KaldiMetaData()
    self._classes = {}

    logging.info('Loading meta data ...')
    self.load_meta_data()

    if self.mode == utils.INFER and self.whole_utt_inference:
      # Do whole utterance inference.
      logging.info('Set chunk_size = 10M and padding = False.')
      self.sampler = ChunkSampler(self.meta, 10000000)
      self.sampler.pad_chunks = False
    else:
      self.sampler = ChunkSampler(self.meta, self.chunk_size_frames)

    if self.mode != utils.INFER and self.add_random_offset:
      self.sampler.add_random_offset = True

    if self.mode != utils.INFER and self.drop_short_chunks > 0.0:
      logging.info('Dropping chunks < %f .' % (self.drop_short_chunks))
      self.sampler.drop_short_chunks = self.drop_short_chunks

    if self.mode != utils.INFER:
      logging.info('Single chunk sampling enabled.')
      self.sampler.single_chunk = self.single_chunk

    if self.mode == utils.TRAIN:
      logging.info('Utt selection by spk enabled for training.')
      self.sampler.select_by_spk = self.select_by_spk_train
    elif self.mode == utils.EVAL:
      logging.info('Utt selection by spk enabled for evaluation.')
      self.sampler.select_by_spk = self.select_by_spk_eval

    if self.mode == utils.TRAIN:
      logging.info('Num repeats = %d.' % (self.num_repeats))
      self.sampler.num_repeats = self.num_repeats

  def load_meta_data(self):
    ''' Load meta data. '''
    data_paths = self.dataconf[self.mode]['paths']
    logging.info('Loading mode %s dirs: %s ...' % (self.mode, data_paths))
    if len(data_paths) != 1:
      raise ValueError('More than 1 data dirs is not supported by now.')
    for data_path in data_paths:
      logging.info('Loading dir %s ...' % (data_path))
      if self.data_type == 'KaldiDataDirectory':
        self.meta.load(data_path)
      else:
        raise ValueError('Unsupported data type: %s' % (self.data_type))
    self._classes = self.meta.spk2id

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

  def generate_cmvn(self, filelist=None, dry_run=False):  # pylint: disable=unused-argument
    ''' Generate mean and vars of features. '''
    sums, square, count = utils.create_cmvn_statis(
        self.taskconf['audio']['feature_size'],
        self.taskconf['audio']['add_delta_deltas'])

    self.sampler.chunk_size = 100000
    self.sampler.pad_chunks = False

    num_done = 0
    for inputs, _, _, _, _ in \
        self.generate_data():
      # update stats
      if inputs.ndim == 3:
        inputs = np.expand_dims(inputs, axis=0)
      sums, square, count = utils.update_cmvn_statis(
          inputs, sums, square, count, axis=(0, 1))
      num_done += 1
      if num_done % 100 == 0:
        logging.info('Done %d samples.' % (num_done))
      if num_done > self.cmvn_max_samples:
        break
    # compute cmvn
    mean, var = utils.compute_cmvn(sums, square, count)
    if dry_run:
      logging.info('save cmvn:{}'.format(self._cmvn_path))
    else:
      np.save(self._cmvn_path, (mean, var))
    logging.info('generate cmvn done')
    logging.info(mean)
    logging.info(var)

  #pylint: disable=stop-iteration-return
  def generate_data(self):
    '''
    Yields samples.

    Args:
      multiprocess: use multiprocessing. Default = True.

    Yields:
      (inputs, label, filename, clip_id, soft_label)
    '''
    class_num = self.taskconf['classes']['num']

    def process_sample(sample, clip_id):
      '''
      Pack various info into a tuple, to be furtherly processed by Dataset.

      Args:
        sample: a clip (or chunk) of an utterance generated previously.
        clip_id: the index of clip in the entire utterance.

      Returns:
        a tuple of feature, label and everything else for training.
      '''
      inputs, label, utt_key = sample
      filename = utt_key
      clip_id = clip_id
      soft_label = np.zeros((1,))  # disabled for speaker model
      return inputs, label, filename, clip_id, soft_label

    if self.mode == utils.INFER:
      # Estimator.predict might cause multiprocessing to fail.
      multiprocess = False
    else:
      multiprocess = True

    self.sampler.reset()

    if multiprocess:
      q = ImapUnorderedDataQueue
      data_queue = q(self.meta, self.sampler, num_processes=4)
      data_queue.start()
      for samples in data_queue.get_items():
        for idx, sample in enumerate(samples):
          yield process_sample(sample, idx)
    else:
      for item in self.meta.utts.items():
        samples = self.sampler.utt_to_samples((None, item))
        for idx, sample in enumerate(samples):
          yield process_sample(sample, idx)
    raise StopIteration

  def feature_spec(self):
    output_shapes = (
        tf.TensorShape(self.feature_shape),  # audio_feat e.g. (3000, 40, 3)
        tf.TensorShape([]),  # label
        tf.TensorShape([]),  # filename
        tf.TensorShape([]),  # clip_id
        tf.TensorShape([1]),  # soft_label, disabled for speaker model
    )
    output_types = (
        tf.float32,
        tf.int32,
        tf.string,
        tf.int32,
        tf.float32,
    )
    assert len(output_shapes) == len(output_types)
    return output_shapes, output_types

  def preprocess_batch(self, batch):
    return batch

  def dataset(self, mode, batch_size, num_epoch):  # pylint: disable=unused-argument
    shapes, types = self.feature_spec()
    data = tf.data.Dataset.from_generator(
        generator=lambda: self.generate_data(),  # pylint: disable=unnecessary-lambda
        output_types=types,
        output_shapes=shapes,
    )

    buffer_size = self.taskconf['shuffle_buffer_size']
    logging.info('Using buffer size of %d samples in shuffle_and_repeat().' %
                 (buffer_size))
    if mode == utils.TRAIN:
      data = data.shuffle(buffer_size=buffer_size)
      if self.uniform_resample:

        def class_func(inputs, labels, filenames, clip_ids, soft_labels):
          ''' Return the label of a sample tuple. '''
          return labels

        target_dist = tf.ones((self.num_class,), dtype=tf.float32) / \
                      self.num_class
        data = data.apply(
            tf.data.experimental.rejection_resample(class_func, target_dist))

    def make_example(inputs, labels, filenames, clip_ids, soft_labels):
      features = {
          'inputs': inputs,
          'labels': labels,
          'filepath': filenames,
          'clipid': clip_ids,
          'soft_labels': soft_labels,
      }
      return features, labels

    if self.mode == utils.INFER and self.whole_utt_inference:
      # To avoid length difference since padding = False.
      logging.info('Inference mode, set batch_size to 1.')
      batch_size = 1
    return data.map(make_example, num_parallel_calls=10).\
                batch(batch_size, drop_remainder=False).\
                prefetch(tf.data.experimental.AUTOTUNE)

class KaldiDir:

  def __init__(self, kaldi_dir):
    self._dir = Path(kaldi_dir)

    feats_scp_file = self._dir.joinpath('feats.scp')
    spk2utt_file = self._dir.joinpath('spk2utt')
    utt2spk_file = self._dir.joinpath('utt2spk')
    spk2id_file = self._dir.joinpath('spk2id')

    self._feats_scp = defaultdict(str)
    with feats_scp_file.open(mode='r', encoding='utf-8') as scp_reader:
      for line in scp_reader:
        line = line.strip().replace('\n', '').replace('\r',
                                                      '').replace('\t', ' ')
        splits = line.split()
        if len(splits) < 2:
          continue
        utt_id = splits[0]
        utt_path = splits[1]
        self._feats_scp[utt_id] = utt_path
    self._num_utt = len(self._feats_scp)
    logging.info(f"kaldi dir: {self._num_utt} utts")

    self._spk2utt = defaultdict(list)
    with spk2utt_file.open(mode='r', encoding='utf-8') as reader:
      for line in reader:
        spk, utt_str = line.strip().split(maxsplit=1)
        utts = utt_str.split()
        self._spk2utt[spk] = utts

    self._spk2id = defaultdict(lambda: -1)
    with spk2id_file.open(mode='r', encoding='utf-8') as reader:
      for line in reader:
        spk, spk_id = line.strip().split()
        self._spk2id[spk] = spk_id
    self._num_spk = len(self._spk2id)
    logging.info(f"kaldi dir: {self._num_spk} spks")

    self._utt2spk = defaultdict(str)
    self._utt2spk_ids = defaultdict(int)
    with utt2spk_file.open(mode='r', encoding='utf-8') as reader:
      for line in reader:
        utt, spk = line.strip().split()
        self._utt2spk[utt] = spk
        self._utt2spk_ids[utt] = self._spk2id[spk]

  @property
  def num_utts(self):
    return self._num_utt

  @property
  def num_spks(self):
    return self._num_spk

  @property
  def utt2spk(self):
    return self._utt2spk

  @property
  def utt2spkid(self):
    return self._utt2spk_ids

  @property
  def feats_scp(self):
    return self._feats_scp

  @property
  def spk2utt(self):
    return self._spk2utt

  @property
  def spk2id(self):
    return self._spk2id


def rand_segment(feat, segment_length):
  if feat is None:
    return None
  num_frame = feat.shape[0]
  if segment_length < num_frame:
    index = random.randint(0, (num_frame - segment_length) - 1)
    segment_feat = feat[index:index + segment_length, :]
  else:
    segment_feat = feat
    while segment_feat.shape[0] < segment_length:
      segment_feat = np.concatenate([segment_feat, feat], 0)

    seg_len = segment_feat.shape[0]
    elen = seg_len - segment_length - 1
    if elen > 1:
      s = np.round(random.randint(0, elen - 1))
    else:
      s = 0
    e = s + segment_length
    segment_feat = segment_feat[s:e, :]

  return [segment_feat]


def segments(inputs, segment_length, segment_shift_rate=0.5, rand=False):
  '''
  param:
    inputs: [time, feat size, 1]
  return:
    list of feats, shape [segment_len, feat size, 1] or None
  '''
  if rand:
    return rand_segment(inputs, segment_length)

  segment_shift = int(segment_length * segment_shift_rate)
  length = inputs.shape[0]
  segs = []
  for x in range(0, length, segment_shift):
    end = x + segment_length
    if end < length:
      feature_mat = inputs[x:end, :, :]
    else:
      input_data = inputs
      while input_data.shape[0] < segment_length:
        input_data = np.concatenate([input_data, inputs], 0)

      seg_len = input_data.shape[0]
      elen = seg_len - segment_length - 1
      if elen > 1:
        s = np.round(random.randint(0, elen - 1))
      else:
        s = 0
      e = s + segment_length
      feature_mat = input_data[s:e, :, :]
    segs.append(feature_mat)
  return segs


@registers.task.register
class SpeakerUttTask(SpeechTask, tf.keras.utils.Sequence):
  ''' Speaker Task for uttrance with segments '''

  def __init__(self, config, mode):
    super().__init__(config, mode)
    self.shuffle = mode == utils.TRAIN
    self.dataconf = self.config['data']
    self.taskconf = self.dataconf['task']
    self.solverconf = self.config['solver']
    self.batch_size = self.solverconf['optimizer']['batch_size']

    self.data_type = self.taskconf['data_type']
    self.min_segment_length = 240
    self.max_segment_length = 280
    self.segment_shift_rate = 0.5

    self.collect_meta()
    self.on_epoch_end()

  def collect_meta(self):
    data_paths = self.dataconf[self.mode]['paths']
    logging.info('Loading mode: [%s] dirs: %s ...' % (self.mode, data_paths))
    if len(data_paths) != 1:
      raise ValueError('More than 1 data dirs is not supported by now.')

    for data_path in data_paths:
      logging.info('Loading dir %s ...' % (data_path))
      if self.data_type == 'KaldiDataDirectory':
        self.kaldi_meta = KaldiDir(data_path)

        self.feats_scp_items = list(self.kaldi_meta.feats_scp.items())
        self.num_utts = self.kaldi_meta.num_utts
        self.class_nums = self.kaldi_meta.num_spks
        assert self.class_nums == self.taskconf['classes']['num']
        logging.info('The dataset have {} utts and {} speakers'.format(
            self.num_utts, self.class_nums))

        for _, (utt, feat_path) in enumerate(self.kaldi_meta.feats_scp.items()):
          self.feat_size = kaldiio.load_mat(feat_path).shape[1]
          break
        logging.info('feat_size {}'.format(self.feat_size))
        if self.feat_size < 0:
          raise Exception('Wrong feat_size {}'.format(self.feat_size))
      else:
        raise ValueError('Unsupported data type: %s' % (self.data_type))

    self._classes = self.kaldi_meta.spk2id

  def __len__(self):
    ''' the number of exmaples'''
    # using this to make shure the last exmaples less than batch size used.
    if self.mode == utils.TRAIN:
      steps_per_epoch = (self.num_utts - self.batch_size) / self.batch_size + 1
    else:
      steps_per_epoch = (self.num_utts / self.batch_size) + 1
    return int(steps_per_epoch)

  def on_epoch_end(self):
    ''' update indexes after each epoch'''
    self.indexs = np.arange(self.num_utts)
    if self.shuffle:
      logging.info("shuffle data")
      np.random.shuffle(self.indexs)
    logging.info(f"{self.indexs}")

    self.segment_win = int(
        (self.min_segment_length + self.max_segment_length) / 2)
    logging.info(f"default segment length: {self.segment_win}")

    if self.mode == utils.TRAIN:
      if self.min_segment_length <= self.max_segment_length and self.max_segment_length > 0:
        self.segment_win = np.random.randint(self.min_segment_length,
                                             self.max_segment_length)
      logging.info(f"For training, using segment length: {self.segment_win}")

  def collate_fn(self, batch):

    minibatch_size = len(batch)

    inputs = []
    utt_ids = []
    spk_ids = []
    seg_ids = []
    for x in range(minibatch_size):
      sample = batch[x]
      utt_id = sample[0]
      spk_id = int(sample[1])
      spect = sample[2]

      # for train
      for i, spect_slice in enumerate(
          segments(
              spect,
              self.segment_win,
              segment_shift_rate=self.segment_shift_rate,
              rand=self.mode == utils.TRAIN)):
        inputs.append(spect_slice)
        seg_ids.append(i)
        utt_ids.append(utt_id)
        spk_ids.append(spk_id)

    targets = spk_ids
    return utt_ids, seg_ids, inputs, targets

  def __getitem__(self, batch_index, return_format=True):
    ''' get batch_index's batch data '''
    indexs = self.indexs[batch_index * self.batch_size:(batch_index + 1) *
                         self.batch_size]
    # key, feat_path
    batch_meta = [self.feats_scp_items[i] for i in indexs]

    batches = []
    for _, (utt_id, utt_path) in enumerate(batch_meta):
      spk_id = self.kaldi_meta.utt2spkid[utt_id]
      in_feat = kaldiio.load_mat(utt_path)
      in_feat = np.expand_dims(in_feat, axis=-1)
      batches.append((utt_id, spk_id, in_feat))

    uttids, segids, feats, spkids = self.collate_fn(batches)

    if not return_format:
      return uttids, segids, feats, spkids

    labels = np.array(spkids, dtype=np.int32)
    features = {
        'inputs': np.array(feats, dtype=np.float64),
        'labels': labels,
        'filepath': np.array(uttids),
        'clipid': np.array(segids, dtype=np.int32),
    }
    one_hot_labels = tf.keras.utils.to_categorical(
        labels, num_classes=self.class_nums)
    return features, one_hot_labels

  def generate_cmvn(self, filelist=None, dry_run=False):  # pylint: disable=unused-argument
    pass

  def generate_feat(self, filelist, dry_run=False):
    pass

  def preprocess_batch(self, batch):
    return batch

  def generate_data(self):
    '''
    Yields samples.

    Args:
      multiprocess: use multiprocessing. Default = True.

    Yields:
      (inputs, label, filename, clip_id, soft_label)
    '''
    for i in range(len(self)):
      uttids, segids, feats, spkids = self.__getitem__(i, return_format=False)
      for b, _ in enumerate(uttids):
        yield uttids[b], segids[b], feats[b], spkids[b]

    self.on_epoch_end()
    raise StopIteration

  def feature_spec(self):
    output_shapes = (
        tf.TensorShape([]),  # utt
        tf.TensorShape([]),  # segid
        tf.TensorShape([None, self.feat_size,
                        1]),  # audio_feat, without batch dim e.g. (3000, 40, 3)
        tf.TensorShape([]),  # spkid
    )
    output_types = (
        tf.string,
        tf.int32,
        tf.float32,
        tf.int32,
    )
    assert len(output_shapes) == len(output_types)
    return output_shapes, output_types

  def dataset(self, mode, batch_size, num_epoch):  # pylint: disable=unused-argument
    shapes, types = self.feature_spec()
    data = tf.data.Dataset.from_generator(
        generator=lambda: self.generate_data(),  # pylint: disable=unnecessary-lambda
        output_types=types,
        output_shapes=shapes,
    )

    buffer_size = self.taskconf['shuffle_buffer_size']
    logging.info('Using buffer size of %d samples in shuffle_and_repeat().' %
                 (buffer_size))

    if mode == utils.TRAIN:
      data = data.shuffle(buffer_size=buffer_size)

    def make_example(utts, segids, inputs, labels):
      features = {
          'inputs': inputs,
          'labels': labels,
          'filepath': utts,
          'clipid': segids,
      }
      return features, labels

    return data.map(make_example, num_parallel_calls=30).\
                batch(batch_size, drop_remainder=False).\
                prefetch(tf.data.experimental.AUTOTUNE)
