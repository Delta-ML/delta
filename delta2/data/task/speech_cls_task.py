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
''' emotion speech task '''
import ast
import os
import copy
import threading
import itertools
import functools
import random
from collections import defaultdict

import librosa
import numpy as np
import pandas as pd
import delta.compat as tf
from absl import logging
from sklearn.model_selection import train_test_split

from delta import utils
from delta.data import feat as feat_lib
from delta.utils.register import registers
from delta.data.task.base_speech_task import SpeechTask


def _load_text(text_path):
  with open(text_path + '.txt', 'r', encoding='utf-8') as fin:
    text = ' '.join([line.strip() for line in fin.readlines()])
    return text


#pylint: disable=too-many-public-methods,too-many-instance-attributes
@registers.task.register
class SpeechClsTask(SpeechTask):
  ''' Emotion Task '''

  #pylint: disable=too-many-statements
  def __init__(self, config, mode):
    super().__init__(config, mode)
    # for dev, test dataset split
    self._dev_size = 0.0
    self._test_size = 0.0

    #self.dataconf = self.processdataconf(config)
    self.dataconf = self.config['data']
    self.taskconf = self.dataconf['task']
    self.solverconf = self.config['solver']

    self._data_path = self.dataconf[mode]['paths']
    logging.info("data_path : {}".format(self._data_path))
    if 'segments' in self.dataconf[
        mode] and self.dataconf[mode]['segments'] is not None:
      self._textgrid_path = self.dataconf[mode]['segments']
    else:
      self._textgrid_path = []

    self._file_suffix = self.taskconf['suffix']
    assert self._file_suffix in ['.wav', '.npy'], "Only support wav and npy"
    assert isinstance(self._data_path, (tuple, list))
    assert isinstance(self._textgrid_path, (tuple, list))
    assert self._data_path

    self.input_type = 'samples' if self._file_suffix == '.wav' else 'features'
    self._clip_size = self.taskconf['audio']['clip_size']  # seconds
    if mode == utils.TRAIN:
      self._stride = self.taskconf['audio']['stride']  # percent
    else:
      self._stride = 1.0  # percent
    logging.info("Mode: {}, stride {}".format(mode, self._stride))
    self._feature_type = self.taskconf['audio']['feature_extractor']
    self._feature_name = self.taskconf['audio']['feature_name']
    logging.info(
        f"feature type: {self._feature_type}, feature name: {self._feature_name}"
    )
    self._sample_rate = self.taskconf['audio']['sr']
    self._winstep = self.taskconf['audio']['winstep']
    self._feature_size = self.taskconf['audio']['feature_size']
    self._input_channels = 3 if self.taskconf['audio']['add_delta_deltas'] else 1
    self._cmvn_path = self.taskconf['audio']['cmvn_path']
    self._save_feat_path = self.taskconf['audio']['save_feat_path']

    # {class: [filename, duration, class],...}
    self._class_file = defaultdict(list)
    self._num_class = 0
    self._classes = self.taskconf['classes']['vocab']
    assert self._classes

    self._max_text_len = self.taskconf['text']['max_text_len']
    self._vocab_path = self.taskconf['text']['vocab_path']
    self._epoch = 0

    self._class_file_train = None
    self._train_meta = None
    self._train_by_filename = None
    self._class_file_dev = None
    self._class_train_dev_test = None
    self._postive_segs_in_sec = None
    self._train_label_example = None
    self._class_file_test = None
    self._postive_segs = None
    self.data_items = None

    # generate segment index
    self.generate_meta(mode)

    # load text vocab table
    use_text = self.taskconf['text']['enable']
    if use_text:
      self.load_text_vocab_table()

    # use distilling
    self.use_distilling = False
    if 'distilling' in self.solverconf:
      self.use_distilling = self.solverconf['distilling']['enable']

    if self.use_distilling:
      temperature = self.solverconf['distilling']['temperature']
      model = self.solverconf['distilling']['teacher_model']
      name = self.solverconf['distilling']['name']
      self.teacher = registers.serving[name](model, None, temperature)

  @property
  def max_text_len(self):
    ''' max length of text'''
    return self._max_text_len

  @property
  def num_class(self):
    ''' num class'''
    return len(self._classes)

  @property
  def classes(self):
    ''' classes for labels'''
    return self._classes

  def class_id(self, cls):
    ''' class id '''
    return self._classes[cls]

  @property
  def pos_id(self):
    ''' postive label id'''
    return self.class_id(self.taskconf['classes']['positive'])

  @property
  def example_len(self):
    ''' samples per clip'''
    samples_per_clip = self._sample_rate * self._clip_size
    return samples_per_clip

  def sample_to_frame(self, samples):
    ''' sample num to frame num'''
    nframe = librosa.core.samples_to_frames(
        samples,
        hop_length=self.taskconf['audio']['winstep'] * self._sample_rate)
    return int(nframe)

  @property
  def nframe(self):
    ''' num frames'''
    return self.sample_to_frame(self.example_len)

  @staticmethod
  def feat_output_shape(config):
    ''' without batch_size'''
    if 'feature_shape' in config['task']['audio'] and config['task']['audio'][
        'feature_shape']:
      return config['task']['audio']['feature_shape']

    if config['task']['suffix'] == '.npy':
      input_channels = 3 if config['task']['audio']['add_delta_deltas'] else 1
      nframe = librosa.time_to_frames(
          config['task']['audio']['clip_size'],
          sr=config['task']['audio']['sr'],
          hop_length=config['task']['audio']['winstep'] *
          config['task']['audio']['sr'])
      feature_shape = [
          nframe, config['task']['audio']['feature_size'], input_channels
      ]
    else:
      feature_shape = [
          config['task']['audio']['sr'] * config['task']['audio']['clip_size']
      ]
    config['task']['audio']['feature_shape'] = feature_shape
    return feature_shape

  @property
  def feature_shape(self):
    ''' feature shape w/o batch size '''
    return SpeechClsTask.feat_output_shape(self.dataconf)

  @staticmethod
  def _down_sample(feat, skip_num=4, context=(-7, 7)):
    left_ctx, right_ctx = context

    left = np.tile(feat[0], [abs(left_ctx), 1, 1])
    right = np.tile(
        feat[-1],
        [min(abs(right_ctx),
             abs(right_ctx) - feat.shape[0] % skip_num), 1, 1])
    context = np.concatenate((left, feat, right), axis=0)

    index = np.array([i for i in range(feat.shape[0]) if i % skip_num == 0
                     ]) + abs(left_ctx)
    res = [context[i + left_ctx:i + right_ctx + 1] for i in index]
    return res

  #pylint: disable=arguments-differ
  def generate_feat(self, filelist, dry_run=False):
    '''
    collect *.wav under `paths` and generate *.npy,
       withoud consider segments file
    filelist: paths: list, path of wav data
    '''
    featconf = self.taskconf['audio'].update({'dry_run': dry_run}) \
      if dry_run else self.taskconf['audio']
    logging.debug("feat config: {}".format(featconf))

    files = []
    threads = []
    for data_path in filelist:
      for root, dirname, filenames in os.walk(data_path):
        del dirname
        for filename in filenames:
          if filename.endswith('.wav'):
            filename = os.path.join(root, filename)
            files.append(filename)

      if self._feature_type == 'tffeat':
        func = feat_lib.extract_feature
        featconf.update({'feature_name': self._feature_name})
      elif self._feature_type == 'pyfeat':
        func = feat_lib.extract_feat
      else:
        raise ValueError("Not support feat: {}".format(self._feature_type))

      #pylint: disable=invalid-name
      t = threading.Thread(
          target=func, args=tuple(files), kwargs=featconf, daemon=True)
      threads.append(t)
      t.start()
      files = []

    for t in threads:  #pylint: disable=invalid-name
      t.join()
    logging.info('generate feature done')

  #pylint: disable=arguments-differ,too-many-locals
  def generate_cmvn(self, filelist=None, dry_run=False):
    del filelist
    assert self._stride == 1.0
    batch_size = self.config['solver']['optimizer']['batch_size']
    features, labels = self.input_fn(
        utils.INFER, batch_size,
        num_epoch=1)().make_one_shot_iterator().get_next()
    del labels

    suffix = self.taskconf['suffix']
    if suffix == '.npy':
      logging.info('generate cmvn from numpy')
      feature = features['inputs']
    else:
      logging.info('genearte cmvn from wav')
      # tf extractor graph
      params = feat_lib.speech_ops.speech_params(
          sr=self.taskconf['audio']['sr'],
          bins=self.taskconf['audio']['feature_size'],
          add_delta_deltas=self.taskconf['audio']['add_delta_deltas'],
          audio_frame_length=self.taskconf['audio']['winlen'],
          audio_frame_step=self.taskconf['audio']['winstep'])

      #[batch, Time] -> [batch, time, audio_channel]
      waveforms = tf.expand_dims(features['inputs'], axis=-1)
      #[batch, Time, feat_size, channles]
      feature = feat_lib.speech_ops.batch_extract_feature(waveforms, params)

    # create stats vars
    sums, square, count = utils.create_cmvn_statis(
        self.taskconf['audio']['feature_size'],
        self.taskconf['audio']['add_delta_deltas'])
    try:
      with tf.Session() as sess:
        while True:
          feat_np = sess.run(feature)
          # update stats
          sums, square, count = utils.update_cmvn_statis(
              feat_np, sums, square, count, axis=(0, 1))
    except tf.errors.OutOfRangeError:
      pass

    # compute cmvn
    mean, var = utils.compute_cmvn(sums, square, count)
    logging.info('mean:{}'.format(mean))
    logging.info('var:{}'.format(var))
    if not dry_run:
      np.save(self._cmvn_path, (mean, var))
    logging.info('save cmvn:{}'.format(self._cmvn_path))
    logging.info('generate cmvn done')

  def get_duration(self, filename, sr):  #pylint: disable=invalid-name
    ''' time in second '''
    if filename.endswith('.npy'):
      nframe = np.load(filename).shape[0]
      return librosa.frames_to_time(
          nframe, hop_length=self._winstep * sr, sr=sr)

    if filename.endswith('.wav'):
      return librosa.get_duration(filename=filename)

    raise ValueError("filename suffix not .npy or .wav: {}".format(
        os.path.splitext(filename)[-1]))

  def get_class_files_duration(self):
    ''' dirnames under dataset is class name
     all data_path have same dirnames '''
    classes = None
    for root, dirnames, filenames in os.walk(self._data_path[0]):
      classes = dirnames
      break

    assert classes, 'can not acsess {}'.format(self._data_path[0])
    assert set(classes) == set(self._classes.keys()), '{} {}'.format(
        classes, self._classes.keys())

    def _get_class(path):
      ret = None
      for cls in self._classes:
        if cls in path:
          ret = cls
      return ret

    # to exclude some data under some dir
    excludes = []
    #pylint: disable=too-many-nested-blocks
    for data_path in self._data_path:
      logging.debug("data path: {}".format(data_path))
      for root, dirname, filenames in os.walk(data_path):
        del dirname
        for filename in filenames:
          if filename.endswith(self._file_suffix):
            class_name = _get_class(root)  # 'conflict' or 'normal' str
            assert class_name is not None
            filename = os.path.join(root, filename)

            if excludes:
              for exclude in excludes:
                if exclude in filename:
                  pass

            duration = self.get_duration(
                filename=filename, sr=self._sample_rate)
            self._class_file[class_name].append(
                (filename, duration, class_name))
          else:
            pass

    if not self._class_file:
      logging.debug("class file: {}".format(self._class_file))
      logging.warn("maybe the suffix {} file not exits".format(
          self._file_suffix))

  def save_csv_files_duration(self, save_file='./filelist.csv', force=False):
    ''' save meta data to csv '''
    #pylint: disable=invalid-name
    if not os.path.exists(save_file) or force:
      df = pd.DataFrame.from_dict(self._class_file, orient='index')
      df = df.transpose()
      #logging.info(df)
      df.to_csv(save_file, sep=' ')

  def split_train_dev_test(self, dev_size=0.1, test_size=0.1):
    ''' split data to train, dev, test set'''
    self._class_file_train = []
    self._class_file_dev = []
    self._class_file_test = []
    self._class_train_dev_test = defaultdict(tuple)

    def _split_train_dev_test(filelist, dev_size, test_size, dummpy=True):
      if dummpy:
        return filelist, [], []

      train_file, test_file = train_test_split(filelist, test_size=test_size)
      train_file, dev_file = train_test_split(train_file, test_size=dev_size)
      return train_file, dev_file, test_file

    # class file: dict: key `class`, value: [filename, duration, class]
    nclass = len(self._class_file)
    try:
      sub_dev_size = dev_size / nclass
      sub_test_size = test_size / nclass
    except ZeroDivisionError:
      logging.info("{}".format(self._class_file.keys()))

    for cls in self._class_file.keys():  # class
      train_file, dev_file, test_file = \
        _split_train_dev_test(self._class_file[cls], sub_dev_size, sub_test_size)

      self._class_train_dev_test[cls] = (train_file, dev_file, test_file)

      self._class_file_train.extend(train_file)
      self._class_file_dev.extend(dev_file)
      self._class_file_test.extend(test_file)

    #logging.info(self._class_train_dev_test)
    logging.debug('train: {}'.format(self._class_file_train))
    logging.debug('dev: {}'.format(self._class_file_dev))
    logging.debug('test: {}'.format(self._class_file_test))

  def class_features(self, segments_path):
    ''' map filename w/o suffix to segment samples
       {hxx/xxx/abc:[ [200, 300], ...]}
    '''
    self._postive_segs = defaultdict(list)
    self._postive_segs_in_sec = defaultdict(list)

    #pylint: disable=invalid-name
    def seconds_to_samples(segs_list, pos_example=True, sr=8000):
      '''
            Args:
               segs_list: list of tuple (min, max), unit seconds
               pos_example:
                  if True, POS segments in Positive sentence
                  else, NEG segments in Positive sentence
            Return:
            segs in samples
            '''

      def _convert(second_range):
        ''' second_range, tuple(min, max), unit sencods '''
        low, hight = list(
            map(
                functools.partial(librosa.time_to_samples, sr=sr),
                second_range))
        return (low, hight)

      segs_list = list(map(_convert, segs_list))

      assert pos_example, 'Now only use POS segment in POSITIVE sentece'
      if pos_example:
        # positive segment in positive sentence
        return segs_list

      # negtive segment in positive sentence
      res = []
      flat_list = [0] + list(itertools.chain.from_iterable(segs_list)) + [None]
      for _ in range(0, len(flat_list), 2):
        #if flat_list[0] != flat_list[1]:
        #pylint: disable=invalid-name
        s, e = flat_list[0], flat_list[1]
        if s != e:
          res.append((s, e))
      return res

    def load_segments(segments_file):
      ''' load TextGrid file '''
      if self._postive_segs:
        logging.debug('TextGrid has been loaded: {}'.format(segments_file))
        return

      if segments_file:
        logging.debug('TextGrid loading: {}'.format(segments_file))
        for segment_file in segments_file:
          with open(segment_file) as f:  #pylint: disable=invalid-name
            for line in f:
              filename, segs = line.split(maxsplit=1)
              # using abspath of filename, without file suffix
              filename = os.path.splitext(filename)[0]
              segs = list(map(ast.literal_eval, segs.split()))
              self._postive_segs_in_sec[filename].extend(segs)
              # second -> sample
              segs = seconds_to_samples(segs)
              self._postive_segs[filename].extend(segs)
        #logging.info(self._postive_segs)

    def extract_segments(filename, segments_file):
      '''
            Brief:
              extract segments from filename with TextGrid file
              if `filename` in TextGrid file, then extract segments
              else load hole file samples
            Args:
              filename: wavfile name, abspath
              segments_file: TextGrid path
            Return :
                list of segments index (min, max), unit sample
            '''
      # load TextGrid file
      load_segments(segments_file)

      seg_flag = False  # whther using segment or whole file
      segs = []
      # using abspath without file suffix
      filename = os.path.splitext(filename)[0]
      if filename in self._postive_segs:
        # segement wave file
        # now only positive example has TextGrid file
        segs_index = self._postive_segs[filename]
        segs.extend(segs_index)
        seg_flag = True
      else:
        # return whole file
        segs.append((0, None))
        seg_flag = False
      logging.debug("filename: {} segs_index:{} seg_or_whole:{}".format(
          filename, segs, seg_flag))
      return segs, seg_flag

    #pylint: disable=too-many-locals,invalid-name
    def segment_indexes(filename,
                        segments_file,
                        clip_size=3,
                        stride=0.5,
                        sr=8000):
      ''' generate clip index in samples '''
      duration = self.get_duration(filename=filename, sr=self._sample_rate)
      clip_samples, stride_samples, total_samples = librosa.time_to_samples(
          [clip_size, clip_size * stride, duration], sr=sr)

      segs_index, _ = extract_segments(filename, segments_file)
      seg_index = []

      #pylint: disable=invalid-name
      for st, ed in segs_index:
        if ed is None:
          ed = librosa.time_to_samples(duration, sr=sr)
          logging.debug('file end: {} {}'.format(duration, ed))
        seg_len = ed - st

        sub_segs = []
        if seg_len > clip_samples:
          nstride = int((seg_len - clip_samples) / stride_samples + 1)
          # (strart, end, npad), only support right hand pad
          sub_segs = [(i * stride_samples, i * stride_samples + clip_samples, 0)
                      for i in range(nstride)]
          seg_index.extend(sub_segs)
        else:
          gap = clip_samples - seg_len
          if st - gap >= 0:
            assert ed - (st - gap) == clip_samples
            sub_segs.append((st - gap, ed, 0))
          if ed + gap <= total_samples:
            assert ed + gap - st == clip_samples
            sub_segs.append((st, ed + gap, 0))
          else:
            npad = ed + gap - total_samples
            assert ed + gap - st == clip_samples
            sub_segs.append((st, ed + gap, npad))
          seg_index.extend(sub_segs)
        logging.debug('Input segs: {}, Output segs: {}'.format((st, ed),
                                                               sub_segs))
      return seg_index

    def _generate_online():
      ''' generate clips of file '''
      class_file = self._class_file_train
      # file, duration, class
      for filename, duration, label in class_file:
        del duration
        # [ (start, end, npad)...]
        seg_idxs = segment_indexes(
            filename,
            segments_path,
            clip_size=self._clip_size,
            stride=self._stride,
            sr=self._sample_rate)

        for clip_id, seg in enumerate(seg_idxs):
          example = (filename, label, seg, clip_id)
          self._train_meta.append(example)
          self._train_label_example[label].append(example)

    self._train_meta = []
    self._train_label_example = defaultdict(list)

    _generate_online()

  def over_sampling(self, datatype='train', ratio=1.0):
    ''' over sampling
    prams: ratio: float, adjust of positive example by ratio when do two classes task
    '''
    assert datatype == 'train'

    def _balance(raw_labels):
      labels = copy.deepcopy(raw_labels)
      labels_len = {k: len(v) for k, v in raw_labels.items()}
      logging.info('label_len: {}'.format(labels_len))
      logging.info('ratio: {}'.format(ratio))

      maxlen = max(labels_len.values())
      gaps = {k: maxlen - v for k, v in labels_len.items()}
      logging.info('gaps need: {}'.format(gaps))

      if self.taskconf['classes']['num'] == 2:
        #for two classes, adjust postive example weight by ratio
        pos_len = int(maxlen * abs(ratio - 1))
        maxlen = maxlen + pos_len
        for k in gaps:
          if k == self.taskconf['classes']['positive']:
            gaps[k] += pos_len
        logging.info('gaps by ratio: {}'.format(gaps))

      for k in raw_labels:
        raw_len = int(len(raw_labels[k]))
        if gaps[k] > raw_len:
          for _ in range(int(gaps[k] / raw_len)):
            indexs = np.random.choice(raw_len, size=raw_len, replace=False)
            examples = [raw_labels[k][i] for i in indexs]
            labels[k].extend(examples)
          gaps[k] %= raw_len
        indexs = np.random.choice(raw_len, size=gaps[k], replace=False)
        examples = [raw_labels[k][i] for i in indexs]
        labels[k].extend(examples)

      return maxlen, labels

    maxlen, labels = _balance(self._train_label_example)
    assert all([len(v) <= maxlen for _, v in labels.items()])
    self._train_meta = list(itertools.chain(*labels.values()))

  def collect_by_filename(self):
    ''' collect metadata: filepath -> (label, seg, clip_id)'''
    self._train_by_filename = defaultdict(list)
    for _, (filename, label, seg, clip_id) in enumerate(self._train_meta):
      self._train_by_filename[filename].append((label, seg, clip_id))

  def load_text_vocab_table(self):
    ''' load vocab '''
    if not self._vocab_path and not os.path.exists(self._vocab_path):
      logging.info('vocab_path is not exist.')
      return
    self.word2id = {}
    with open(self._vocab_path, 'r', encoding='utf-8') as vocab_f:
      for line in vocab_f:
        word, idx = line.strip().split()
        self.word2id[word] = idx

  def generate_meta(self, mode):
    ''' collection examples metadata'''
    assert mode in [utils.TRAIN, utils.EVAL, utils.INFER]
    logging.info("{} data".format(mode))
    logging.info("Get classes and coresspoding wavfile")
    self.get_class_files_duration()

    logging.info("Split train dev test dataset ")
    self.split_train_dev_test(
        dev_size=self._dev_size, test_size=self._test_size)

    logging.info("Cut clips and extract features ")
    self.class_features(segments_path=self._textgrid_path)
    if utils.TRAIN in mode:
      logging.info("Blance classes exmales of train dataset ")
      self.over_sampling()

    logging.info("Collection by filepath")
    self.collect_by_filename()
    self.data_items = list(self._train_by_filename.items())

  def shuffle(self):
    ''' oversampling and get examples list meta'''
    logging.info("Blance classes exmales of train dataset ")
    self.over_sampling()

    logging.info("Collection by filepath")
    self.collect_by_filename()

    self.data_items = list(self._train_by_filename.items())

  def _word_table_lookup(self, text):
    ''' convert text to id'''
    max_text_len = self._max_text_len
    text2id = np.zeros(shape=[max_text_len])
    pad_len = min(max_text_len, len(text))
    for char_num in range(pad_len):
      ## handle unk
      if text[char_num] in self.word2id:
        text2id[char_num] = self.word2id[text[char_num]]
      else:
        text2id[char_num] = self.word2id['<unk>']
    return text2id

  #pylint: disable=too-many-statements,too-many-locals,too-many-branches
  def generate_data(self):
    ''' generate one example'''
    use_text = self.taskconf['text']['enable']

    # total files
    total = len(self._train_by_filename.values())
    self._epoch += 1  # epcoh from 1

    batch = []
    np.random.shuffle(self.data_items)
    for i, (filename, examples) in enumerate(self.data_items):
      #logging.info("example info", filename, examples)

      # convert txt to ids
      if use_text:
        text = _load_text('.'.join(filename.split('.')[:-1]))
        text2id = self._word_table_lookup(text)
      else:
        text2id = np.array([0] * self._max_text_len)

      # gen audio or load feat
      if self._file_suffix == '.wav':
        sr, raw_samples = feat_lib.load_wav(filename)  #pylint: disable=invalid-name
        for label, seg, clip_id in examples:
          # examples of one file
          samples = raw_samples
          if seg[2]:
            samples = np.pad(samples, [0, seg[2]], mode='constant')
          samples = samples[seg[0]:seg[1]]
          assert len(samples) == self.example_len, "{} {}".format(filename, seg)

          labelid = self.class_id(label)

          if self.use_distilling:
            raise ValueError("Not Support distilation for *.wav input")
          else:
            class_num = self.taskconf['classes']['num']
            soft_label = [0] * class_num

          if use_text:
            if clip_id == 0:
              # only add into batch when meet the first clip
              batch.append(
                  (samples, text2id, labelid, filename, clip_id, soft_label))
          else:
            batch.append(
                (samples, text2id, labelid, filename, clip_id, soft_label))

      else:
        feat = np.load(filename)

        # shape : [nframe, feat_size, 3]
        if self._feature_type:
          fbank = feat_lib.add_delta_delta(feat, self._feature_size, order=2)
          if self._input_channels == 1:
            fbank = fbank[:, :, 0:1]
        else:
          fbank = feat_lib.delta_delta(feat)

        for label, seg, clip_id in examples:
          feat = fbank
          #logging.info("feat shape: {}".format(feat.shape))

          seg = list(map(self.sample_to_frame, seg))
          if seg[2]:
            # need padding
            feat = np.pad(feat, [(0, seg[2]), (0, 0), (0, 0)], mode='constant')

          feat = feat[seg[0]:seg[1], :, :]
          expect_nframes = self.sample_to_frame(self.example_len)
          if len(feat) != expect_nframes:
            logging.warn("{} {} {} {} {} {}".format(
                filename, seg, len(feat), self.example_len,
                self.sample_to_frame(self.example_len), seg[2]))
            feat = np.pad(
                feat, [(0, expect_nframes - len(feat)), (0, 0), (0, 0)],
                mode='constant')

          if self.use_distilling:
            soft_label = self.teacher(feat)
          else:
            class_num = self.taskconf['classes']['num']
            soft_label = [0] * class_num

          # convert string label to int label
          labelid = self.class_id(label)

          if use_text:
            if clip_id == 0:
              # only add into batch when meet the first clip
              batch.append(
                  (feat, text2id, labelid, filename, clip_id, soft_label))
          else:
            batch.append(
                (feat, text2id, labelid, filename, clip_id, soft_label))

      #if i % 100000:
      #  logging.info('epoch:{} iter exmaple:{} total:{} : {:.2f}%'.format(
      #     self._epoch, i, total, i * 100 / total))

      for inputs, texts, label, filepath, clip_id, soft_label in batch:
        yield inputs, texts, label, filepath, clip_id, soft_label

      batch.clear()

    logging.info("Out of range")
    raise StopIteration  #pylint: disable=stop-iteration-return

  def feature_spec(self):
    class_num = self.taskconf['classes']['num']
    if self.input_type == 'samples':
      # wavforms
      output_shapes = (
          tf.TensorShape([self.example_len]),
          tf.TensorShape([self.max_text_len]),
          tf.TensorShape([]),
          tf.TensorShape([]),
          tf.TensorShape([]),
          tf.TensorShape([class_num]),  # soft_label
      )
    else:
      # features
      output_shapes = (
          tf.TensorShape(self.feature_shape),  # audio_feat (3000, 40, 3)
          tf.TensorShape([self.max_text_len]),  # text
          tf.TensorShape([]),  # label
          tf.TensorShape([]),  # filename
          tf.TensorShape([]),  # clip_id
          tf.TensorShape([class_num]),  # soft_label
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

  #pylint: disable=arguments-differ
  def dataset(self, mode, batch_size, num_epoch):
    shapes, types = self.feature_spec()
    ds = tf.data.Dataset.from_generator(  #pylint: disable=invalid-name
        generator=lambda: self.generate_data(),  #pylint: disable=unnecessary-lambda
        output_types=types,
        output_shapes=shapes,
    )

    if mode == utils.TRAIN:
      ds = ds.apply(  #pylint: disable=invalid-name
          tf.data.experimental.shuffle_and_repeat(
              buffer_size=100 * batch_size, count=num_epoch, seed=None))

    #pylint: disable=too-many-arguments
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

    return ds.apply(
        tf.data.experimental.map_and_batch(
            make_example, batch_size,
            drop_remainder=False)).prefetch(tf.data.experimental.AUTOTUNE)


@registers.task.register
class IEmoCapTask(SpeechClsTask, tf.keras.utils.Sequence):
  ''' iemocap dataset '''

  def __init__(self, config, mode):
    super().__init__(config, mode)
    self.shuffle = mode == utils.TRAIN
    self.batch_size = self.config['solver']['optimizer']['batch_size']
    subset = self.config['data']['task']['subset']
    subset = subset if subset else 'all'
    self.subset = subset
    assert self.subset in ('impro', 'script', 'all')
    logging.info(f"using subset data: {self.subset}, shuffle: {self.shuffle}")

    self.examples_meta = []
    for _, (filename, examples) in enumerate(self.data_items):
      for label, seg, clip_id in examples:
        if clip_id == 0:
          if self.subset in filename or self.subset == 'all':
            # impro script
            self.examples_meta.append((filename, label, seg))

    self.num_examples = len(self.examples_meta)
    logging.info(f"num examples: {self.num_examples}")

    self.on_epoch_end()

  def __len__(self):
    ''' the number of examples '''
    if self.mode == utils.TRAIN:
      steps_per_epoch = (len(self.examples_meta) -
                         self.batch_size) / self.batch_size + 1
    else:
      steps_per_epoch = len(self.examples_meta) / self.batch_size + 1
    return int(steps_per_epoch)

  def on_epoch_end(self):
    ''' update indexes after each epoch'''
    self.indexes = np.arange(len(self.examples_meta))
    if self.shuffle:
      np.random.shuffle(self.indexes)

  #pylint: disable=too-many-locals
  def __getitem__(self, batch_index):
    ''' get batch_index's batch data '''
    assert self._file_suffix == '.npy'

    logging.debug(f'''
        batch_index: {batch_index},
        num_batches: {len(self)},
        num exapmples: {self.num_examples},
        label dict: {self.classes}''')
    indexes = self.indexes[batch_index * self.batch_size:(batch_index + 1) *
                           self.batch_size]
    #logging.info(f"examples meta: {self.examples_meta}")
    batch_meta = [self.examples_meta[i] for i in indexes]

    if self.shuffle:
      random.shuffle(batch_meta)

    logging.debug(f"batch metah: {batch_meta}")
    feats = []
    labels = []
    filenames = []
    for _, (filename, label, seg) in enumerate(batch_meta):
      feat = np.load(filename)

      # shape : [nframe, feat_size, 3]
      feat = feat_lib.add_delta_delta(feat, self._feature_size, order=2)
      if self.feature_shape[-1] == 1:
        feat = feat[:, :, 0:1]

      seg = list(map(self.sample_to_frame, seg))
      if seg[2]:
        # need padding
        feat = np.pad(feat, [(0, seg[2]), (0, 0), (0, 0)], mode='constant')

      feat = feat[seg[0]:seg[1], :, :]
      assert len(feat) == self.sample_to_frame(
          self.example_len), "{} {} {} {} {} {}".format(
              filename, seg, len(feat), self.example_len,
              self.sample_to_frame(self.example_len), seg[2])

      # convert string label to int label
      labelid = self.class_id(label)

      feats.append(feat)
      filenames.append(filename)
      labels.append(labelid)

    features = {
        'inputs': np.array(feats, dtype=np.float64),
        'labels': np.array(labels, dtype=np.int32),
    }

    one_hot_label = np.array(labels, dtype=np.int32)
    one_hot_label = tf.keras.utils.to_categorical(
        one_hot_label, num_classes=len(self.classes))
    return features, one_hot_label

  def batch_input_shape(self):
    ''' batch input TensorShape'''
    feat, label = self.__getitem__(0)
    feat_shape = {}
    for key, val in feat.items():
      feat_shape[key] = tf.TensorShape((None,) + val.shape[1:])

    label_shape = tf.TensorShape((None,) + label.shape[1:])
    return feat_shape, label_shape
