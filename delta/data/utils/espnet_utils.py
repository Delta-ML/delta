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
''' espnet utils'''
import json
from collections import OrderedDict
import numpy as np
from absl import logging

from espnet.utils.cli_readers import file_reader_helper
from espnet.utils.cli_writers import file_writer_helper
from espnet.utils.training.batchfy import make_batchset as make_batchset_espnet
from espnet.utils.io_utils import LoadInputsAndTargets

from delta import utils

TASK_SET = {'asr': 'asr', 'tts': 'tts'}


#pylint: disable=too-many-locals
#pylint: disable=too-many-arguments
def make_batchset(task,
                  data,
                  batch_size,
                  max_length_in,
                  max_length_out,
                  num_batches=0,
                  batch_sort_key='shuffle',
                  min_batch_size=1,
                  shortest_first=False,
                  batch_bins=0,
                  batch_frames_in=0,
                  batch_frames_out=0,
                  batch_frames_inout=0,
                  batch_strategy='auto'):
  """Make batch set from json dictionary

  if utts have "category" value,

     >>> data = {'utt1': {'category': 'A', 'input': ...},
     ...         'utt2': {'category': 'B', 'input': ...},
     ...         'utt3': {'category': 'B', 'input': ...},
     ...         'utt4': {'category': 'A', 'input': ...}}
     >>> make_batchset(data, batchsize=2, ...)
     [[('utt1', ...), ('utt4', ...)], [('utt2', ...), ('utt3': ...)]]

  Note that if any utts doesn't have "category",
  perform as same as batchfy_by_{batch_strategy}

  :param string task: task type, which in [asr, tts]
  :param Dict[str, Dict[str, Any]] data: dictionary loaded from data.json
  :param int batch_size: maximum number of sequences in a minibatch.
  :param int batch_bins: maximum number of bins (frames x dim) in a minibatch.
  :param int batch_frames_in:  maximum number of input frames in a minibatch.
  :param int batch_frames_out: maximum number of output frames in a minibatch.
  :param int batch_frames_out: maximum number of input+output frames in a minibatch.
  :param str batch_strategy: strategy to count maximum size of batch [auto, seq, bin, frame].
      "auto" : automatically detect make_batch strategy by finding enabled args
      "seq"  : create the minibatch that has the maximum number of seqs under batch_size.
      "bin"  : create the minibatch that has the maximum number of bins under batch_bins,
               where the "bin" means the number of frames x dim.
      "frame": create the minibatch that has the maximum number of input, output and input+output
               frames under batch_frames_in, batch_frames_out and batch_frames_inout, respectively

  :param int max_length_in: maximum length of input to decide adaptive batch size
  :param int max_length_out: maximum length of output to decide adaptive batch size
  :param int num_batches: # number of batches to use (for debug)
  :param int min_batch_size: minimum batch size (for multi-gpu)
  :param bool shortest_first: Sort from batch with shortest samples to longest if true,
                              otherwise reverse
  :param str batch_sort_key: how to sort data before creating minibatches [input, output, shuffle]
      :return: List[List[Tuple[str, dict]]] list of batches

  Reference: https://github.com/espnet/espnet/pull/759/files
             https://github.com/espnet/espnet/commit/dc0a0d3cfc271af945804f391e81cd5824b08725
             https://github.com/espnet/espnet/commit/73018318a65d18cf2e644a45aa725323c9e4a0e6
  """

  assert task in list(TASK_SET.keys())

  #swap_io: if True, use "input" as output and "output" as input in `data` dict
  swap_io = False
  if task == TASK_SET['tts']:
    swap_io = True

  minibatches = make_batchset_espnet(
      data,
      batch_size=batch_size,
      max_length_in=max_length_in,
      max_length_out=max_length_out,
      num_batches=num_batches,
      min_batch_size=min_batch_size,
      shortest_first=shortest_first,
      batch_sort_key=batch_sort_key,
      swap_io=swap_io,
      count=batch_strategy,
      batch_bins=batch_bins,
      batch_frames_in=batch_frames_in,
      batch_frames_out=batch_frames_out,
      batch_frames_inout=batch_frames_inout)
  return minibatches


#pylint: disable=too-many-locals
def get_batches(config, mode):
  ''' make batches of metas and get dataset size'''
  assert mode in (utils.TRAIN, utils.EVAL, utils.INFER)

  # read meta of json
  json_path = config['data'][mode]['paths']
  assert len(json_path) == 1
  logging.info(f"=== load json data {json_path} ===")
  logging.info(f" # learning phase: {mode}")

  #pylint: disable=invalid-name
  with open(json_path[0], 'r', encoding='utf-8') as f:
    metas_raw = json.load(f)['utts']

  # sort by utts id
  metas = OrderedDict(sorted(metas_raw.items(), key=lambda t: t[0]))

  # dataset size
  utts = len(metas.keys())
  logging.info(' # utts: ' + str(utts))

  # make batchset
  use_sortagrad = config['data']['task']['sortagrad']
  logging.info(f' # sortagrad: {use_sortagrad}')

  task = config['data']['task']['type']
  assert task in list(TASK_SET.keys())
  # using same json for asr and tts task
  if task == TASK_SET['asr']:
    src = 'src'
    tgt = 'tgt'
  elif task == TASK_SET['tts']:
    src = 'tgt'
    tgt = 'src'
  else:
    raise ValueError("task type must int : {} get : {}".format(
        list(TASK_SET.keys()), task))
  logging.info(f" # task: {task}")

  # delta config
  maxlen_src = config['data']['task'][src]['max_len']
  maxlen_tgt = config['data']['task'][tgt]['max_len']
  batch_sort_key = config['data']['task']['batch_sort_key']
  num_batches = config['data']['task']['num_batches']  # for debug
  global_batch_size = config['solver']['optimizer']['batch_size']
  batch_bins = config['data']['task']['batch']['batch_bins']
  batch_frames_in = config['data']['task']['batch']['batch_frames_in']
  batch_frames_out = config['data']['task']['batch']['batch_frames_out']
  batch_frames_inout = config['data']['task']['batch']['batch_frames_inout']
  batch_strategy = config['data']['task']['batch']['batch_strategy']

  _, ngpu = utils.gpu_device_names()
  min_batch_size = ngpu if ngpu else 1
  logging.info(f" # ngpu: {ngpu}")
  logging.info(f" # min_batch_size: {min_batch_size}")

  minibatches = make_batchset(
      task=task,
      data=metas,
      batch_size=global_batch_size,
      max_length_in=maxlen_src,
      max_length_out=maxlen_tgt,
      num_batches=num_batches,
      batch_sort_key=batch_sort_key,
      min_batch_size=min_batch_size,
      shortest_first=use_sortagrad,
      batch_bins=batch_bins,
      batch_frames_in=batch_frames_in,
      batch_frames_out=batch_frames_out,
      batch_frames_inout=batch_frames_inout,
      batch_strategy=batch_strategy)

  return {'data': minibatches, 'n_utts': utts}


class Converter:
  '''custom batch converter for kaldi
  :param int subsampling_factor: The subsampling factor
  :param object preprocess_conf: The preprocessing config
  '''

  def __init__(self, config):
    self._config = config
    self.subsampling_factor = None
    self.preprocess_conf = None
    self.load_inputs_and_targets = lambda x: x

  @property
  def config(self):
    ''' candy _config'''
    return self._config

  def transform(self, item):
    ''' load inputs and outputs '''
    return self.load_inputs_and_targets(item)

  def __call__(self, batch):
    ''' Transforms a batch
    param: batch, list of (uttid, {'input': [{...}], 'output': [{...}]})
    '''
    # batch should be located in list
    # list of examples
    (xs, ys), uttid_list = self.transform(batch)  #pylint: disable=invalid-name

    # perform subsampling
    if self.subsampling_factor > 1:
      xs = [x[::self.subsampling_factor, :] for x in xs]  #pylint: disable=invalid-name

    # get batch of lengths of input and output sequences
    ilens = [x.shape[0] for x in xs]
    olens = [y.shape[0] for y in ys]

    return xs, ilens, ys, olens, uttid_list


class ASRConverter(Converter):
  ''' ASR preprocess '''

  def __init__(self, config):
    super().__init__(config)
    taskconf = self.config['data']['task']
    assert taskconf['type'] == TASK_SET['asr']
    self.subsampling_factor = taskconf['src']['subsampling_factor']
    self.preprocess_conf = taskconf['src']['preprocess_conf']
    # mode: asr or tts
    self.load_inputs_and_targets = LoadInputsAndTargets(
        mode=taskconf['type'],
        load_output=True,
        preprocess_conf=self.preprocess_conf)

  #pylint: disable=arguments-differ
  #pylint: disable=too-many-branches
  def transform(self, batch):
    """Function to load inputs, targets and uttid from list of dicts

    :param List[Tuple[str, dict]] batch: list of dict which is subset of
        loaded data.json
    :return: list of input token id sequences [(L_1), (L_2), ..., (L_B)]
    :return: list of input feature sequences
        [(T_1, D), (T_2, D), ..., (T_B, D)]
    :rtype: list of float ndarray
    :return: list of target token id sequences [(L_1), (L_2), ..., (L_B)]
    :rtype: list of int ndarray
    Reference: Espnet source code, /espnet/utils/io_utils.py
               https://github.com/espnet/espnet/blob/master/espnet/utils/io_utils.py
    """
    x_feats_dict = OrderedDict()  # OrderedDict[str, List[np.ndarray]]
    y_feats_dict = OrderedDict()  # OrderedDict[str, List[np.ndarray]]
    uttid_list = []  # List[str]

    mode = self.load_inputs_and_targets.mode
    for uttid, info in batch:
      uttid_list.append(uttid)

      if self.load_inputs_and_targets.load_input:
        # Note(kamo): This for-loop is for multiple inputs
        for idx, inp in enumerate(info['input']):
          # {"input":
          #  [{"feat": "some/path.h5:F01_050C0101_PED_REAL",
          #    "filetype": "hdf5",
          #    "name": "input1", ...}], ...}

          #pylint: disable=protected-access
          x_data = self.load_inputs_and_targets._get_from_loader(
              filepath=inp['feat'], filetype=inp.get('filetype', 'mat'))
          x_feats_dict.setdefault(inp['name'], []).append(x_data)

      elif mode == 'tts' and self.load_inputs_and_targets.use_speaker_embedding:
        for idx, inp in enumerate(info['input']):
          if idx != 1 and len(info['input']) > 1:
            x_data = None
          else:
            x_data = self.load_inputs_and_targets._get_from_loader(  #pylint: disable=protected-access
                filepath=inp['feat'],
                filetype=inp.get('filetype', 'mat'))
          x_feats_dict.setdefault(inp['name'], []).append(x_data)

      if self.load_inputs_and_targets.load_output:
        for idx, inp in enumerate(info['output']):
          if 'tokenid' in inp:
            # ======= Legacy format for output =======
            # {"output": [{"tokenid": "1 2 3 4"}])
            x_data = np.fromiter(
                map(int, inp['tokenid'].split()), dtype=np.int64)
          else:
            # ======= New format =======
            # {"input":
            #  [{"feat": "some/path.h5:F01_050C0101_PED_REAL",
            #    "filetype": "hdf5",
            #    "name": "target1", ...}], ...}
            x_data = self.load_inputs_and_targets._get_from_loader(  #pylint: disable=protected-access
                filepath=inp['feat'],
                filetype=inp.get('filetype', 'mat'))

          y_feats_dict.setdefault(inp['name'], []).append(x_data)
    if self.load_inputs_and_targets.mode == 'asr':
      #pylint: disable=protected-access
      return_batch, uttid_list = self.load_inputs_and_targets._create_batch_asr(
          x_feats_dict, y_feats_dict, uttid_list)

    elif self.load_inputs_and_targets.mode == 'tts':
      _, info = batch[0]
      eos = int(info['output'][0]['shape'][1]) - 1
      #pylint: disable=protected-access
      return_batch, uttid_list = self.load_inputs_and_targets._create_batch_tts(
          x_feats_dict, y_feats_dict, uttid_list, eos)
    else:
      raise NotImplementedError

    if self.load_inputs_and_targets.preprocessing is not None:
      # Apply pre-processing only to input1 feature, now
      if 'input1' in return_batch:
        return_batch['input1'] = \
            self.load_inputs_and_targets.preprocessing(return_batch['input1'], uttid_list,
                               **self.load_inputs_and_targets.preprocess_args)

    # Doesn't return the names now.
    return tuple(return_batch.values()), uttid_list
