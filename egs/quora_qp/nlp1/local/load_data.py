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
"""Quora Question Pairs data loader."""
"python local/load_data.py "

import typing
from pathlib import Path

import keras
import pandas as pd
import csv
import sys
from absl import logging


def load_data(url,filepath,stage='train'):
    """
    Load QuoraQP data.

    :param path: `None` for download from quora, specific path for
        downloaded data.
    :param stage: One of `train`, `dev`, and `test`.
    :param task: Could be one of `ranking`, `classification` or a
        :class:`matchzoo.engine.BaseTask` instance.
    :param return_classes: Whether return classes for classification task.
    :return: A DataPack if `ranking`, a tuple of (DataPack, classes) if
        `classification`.
    """

    data_root = _download_data(url,filepath)
    file_path = data_root.joinpath("{}.tsv".format(stage))
  #  data_pack = _read_data(file_path, stage)


def _download_data(url,filepath):
    ref_path = keras.utils.data_utils.get_file(
        'quora_qp', url, extract=True,
        cache_dir=filepath,
        cache_subdir='quora_qp'
    )
    return Path(ref_path).parent.joinpath('QQP')


if __name__=='__main__':
  if sys.argv != 3:
    logging.error("Usage {} input_file output_file".format(sys.argv[0]))
    sys.exit()
  path = Path(sys.argv[1])
  if path.exist():
    url=sys.argv[1]
    load_data(url,file_path=sys.argv[2])
  else:
    logging.error("Path {} is not exit".format(sys.argv[1]))

