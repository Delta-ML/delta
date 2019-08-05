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
""" Meta info for Kaldi data directory. """
import os
from collections import defaultdict

from absl import logging

#pylint: disable=too-many-instance-attributes
#pylint: disable=attribute-defined-outside-init


def _declare_dict_property(cls, name):
  ''' Declare getter and setter property for a given name. '''

  def getter(self):
    return self[name]

  def setter(self, value):
    self[name] = value

  prop = property(fget=getter, fset=setter)
  setattr(cls, name, prop)


def _return_none():
  return None


class Utt(defaultdict):
  ''' An utterance. '''

  def __init__(self, default_factory=None):  # pylint: disable=unused-argument
    super().__init__(_return_none)


LIST_OF_UTT_PROPS = ('wavlen', 'featlen', 'wav', 'feat', 'vad', 'spk')
for one_prop_name in LIST_OF_UTT_PROPS:
  _declare_dict_property(Utt, one_prop_name)


class Spk(defaultdict):
  ''' A speaker. '''

  def __init__(self, default_factory=None):  # pylint: disable=unused-argument
    super().__init__(_return_none)


LIST_OF_UTT_PROPS = ('id', 'utts', 'numutt')
for one_prop_name in LIST_OF_UTT_PROPS:
  _declare_dict_property(Spk, one_prop_name)


class KaldiMetaData():
  ''' Meta data for a kaldi corpus directory. '''

  def __init__(self, extra_files=None):
    self.data_path = None
    self.utts = defaultdict(Utt)
    self.spks = defaultdict(Spk)
    self.files_to_load = [
        ('feats.scp', 'feat'),
        ('feats.len', 'featlen'),
        ('vad.scp', 'vad'),
        ('utt2spk', 'spk'),
    ]
    if extra_files:
      logging.info('Adding extra files to load: %s' % (extra_files))
      self.files_to_load.extend(extra_files)

  @property
  def spk2id(self):
    ''' spk to id list '''
    return {s[0]: s[1].id for s in self.spks.items()}

  def validate(self):
    ''' Sanity check. Make sure everything is (probably) OK. '''
    # TODO: more efficient and robust. Also check speakers.
    for utt_key in self.utts.keys():
      first_utt_key = utt_key
      break
    num_props = len(self.utts[first_utt_key])
    for utt_key, utt in self.utts.items():
      if len(utt) != num_props:
        logging.warning('Utt %s has unequal number of props with %s.' % \
                     (utt_key, first_utt_key))
        return False
      if 'spkid' not in utt:
        utt['spkid'] = self.spks[utt.spk].id
    logging.warning(
        'All utts have same number of props, data dir appears to be OK.')
    return True

  def collect_spks_from_utts(self):
    ''' Convert utt2spk to spk2utt. '''
    for utt_key, utt in self.utts.items():
      if self.spks[utt.spk].utts is None:
        self.spks[utt.spk].utts = []
      self.spks[utt.spk].utts.append(utt_key)
    for spk_idx, spk in enumerate(sorted(self.spks.keys())):
      self.spks[spk].id = spk_idx
      self.spks[spk].numutt = len(self.spks[spk].utts)

  def select_spks(self, spk_list):
    ''' Select speakers and perform a shallow copy. '''
    new_meta = KaldiMetaData()
    for spk in spk_list:
      if spk in self.spks:
        for utt_key in self.spks[spk].utts:
          new_meta.utts[utt_key] = self.utts[utt_key]
    new_meta.collect_spks_from_utts()
    return new_meta

  def select_utts(self, utt_list):
    ''' Select utts and perform a shallow copy. '''
    new_meta = KaldiMetaData()
    for utt_key in utt_list:
      new_meta.utts[utt_key] = self.utts[utt_key]
    new_meta.collect_spks_from_utts()
    return new_meta

  def load(self, data_path):
    ''' Load meta data from Kaldi data directory. '''
    # TODO: support multiple data directories
    logging.info('Loading Kaldi dir: %s ...' % (data_path))
    self.data_path = data_path

    def scp_to_dict(file_path, target_dict, prop):
      '''
      Parse a scp file and target_dict[key][prop] = value.

      Args:
        file_path: str, path to scp file.
        target_dict: the dict to write to.
        prop: prop name used as dict key.

      Return:
        number of lines processed. None if file not found.
      '''
      num_lines = 0
      try:
        with open(file_path, 'r') as fp_in:
          for line in fp_in:
            num_lines += 1
            tokens = line.strip().split(None, 1)
            target_dict[tokens[0]][prop] = tokens[1]
        return num_lines
      except FileNotFoundError:
        return None

    # Load various scp/ark meta files.
    for file_name, prop_name in self.files_to_load:
      file_path = os.path.join(data_path, file_name)
      num_lines = scp_to_dict(file_path, self.utts, prop_name)
      if num_lines:
        logging.info('Loaded file %s (key %s) %d lines.' \
                     % (file_name, prop_name, num_lines))
      else:
        logging.info('File %s (key %s) failed to load.' \
                     % (file_name, prop_name))

    # Reduce by speaker.
    self.collect_spks_from_utts()
    logging.info('Found %d spks.' % (len(self.spks)))

    # Load optional speaker -> id file.
    spk2id_path = os.path.join(data_path, 'spk2id')
    num_lines = scp_to_dict(spk2id_path, self.spks, 'id')
    if num_lines:
      logging.info('Loaded %d spks from spk2id.' % (num_lines))

    # Sanity check.
    self.validate()

  def dump(self, data_path, overwrite=False):
    ''' Write contents to dir in Kaldi style. '''

    # Don't overwrite existing data repo.
    logging.info('Dumping to data dir %s ...' % (data_path))
    if os.path.isdir(data_path) and not overwrite:
      logging.warning('Dir %s exists and overwrite = False, skip.' %
                      (data_path))
      return
    os.makedirs(data_path, mode=0o755, exist_ok=True)

    # Dump data in sorted order.
    sorted_utt_keys = sorted(self.utts.keys())
    for file_name, prop_name in self.files_to_load:
      empty = True
      for utt_key in sorted_utt_keys:
        if self.utts[utt_key][prop_name]:
          empty = False
          break
      if not empty:
        file_path = os.path.join(data_path, file_name)
        logging.info('Dumping prop %s to file %s ...' % (prop_name, file_name))
        with open(file_path, 'w') as fp_out:
          for utt_key in sorted_utt_keys:
            if self.utts[utt_key][prop_name]:
              fp_out.write('%s %s\n' % (utt_key, self.utts[utt_key][prop_name]))

    # Generate speaker -> various data.
    spk_files_to_dump = [
        ('spk2id', {s[0]: s[1].id for s in self.spks.items()}),
        ('spk2utt', {s[0]: s[1].utts for s in self.spks.items()}),
        ('spk2numutt', {s[0]: s[1].numutt for s in self.spks.items()})
    ]
    sorted_spks = sorted(self.spks.keys())
    for file_name, dict_inst in spk_files_to_dump:
      # Maybe we can wrap this into a function?
      logging.info('Dumping file %s ...' % (file_name))
      with open(os.path.join(data_path, file_name), 'w') as fp_out:
        for spk in sorted_spks:
          value = dict_inst[spk]
          if hasattr(value, '__iter__'):
            fp_out.write('%s %s\n' % (spk, ' '.join(sorted(value))))
          else:
            fp_out.write('%s %s\n' % (spk, value))
