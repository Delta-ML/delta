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

"""Data set operation class"""

import os
from typing import List
from absl import logging
from shutil import copyfile, SameFileError

from delta import PACKAGE_ROOT_DIR
from delta.utils.config import load_config
from delta.utils.config import save_config


class BaseDataSet:
  """Base Data set class"""

  def __init__(self, directory: str):

    self.project_dir: str = directory
    self.download_files: List[str] = list()
    self.data_files: List[str] = list()
    self.config_files: List[str] = list()
    self.origin_config_dir = os.path.join(PACKAGE_ROOT_DIR, "configs")

  def download(self):
    """Download dataset from Internet."""
    raise NotImplementedError

  def after_download(self):
    """Dataset operations after download."""
    raise NotImplementedError

  def copy_config_files(self):
    """Copy config files"""
    for config_file in self.config_files:
      full_config_file = os.path.join(self.origin_config_dir, config_file)
      new_config_file = os.path.join(self.config_dir, config_file)
      config = load_config(full_config_file, join_project_dir=False)
      config['data']['project_dir'] = self.project_dir
      save_config(config, new_config_file)

  @property
  def data_dir(self):
    """data directory"""
    return os.path.join(self.project_dir, "data")

  @property
  def config_dir(self):
    """Config directory"""
    return os.path.join(self.project_dir, "config")

  def _download_ready(self):
    """If download is ready."""
    for data_file in self.download_files:
      full_data_file = os.path.join(self.data_dir, data_file)
      if not os.path.exists(full_data_file):
        logging.warning(f"Data: {full_data_file} do not exists!")
        return False
    return True

  def is_ready(self):
    """If the dataset is ready for using."""
    if not os.path.exists(self.project_dir):
      logging.warning(f"Directory: {self.project_dir} do not exists!")
      return False
    for data_file in self.data_files:
      full_data_file = os.path.join(self.data_dir, data_file)
      if not os.path.exists(full_data_file):
        logging.warning(f"Data file: {full_data_file} do not exists!")
        return False
    for config_file in self.config_files:
      full_config_file = os.path.join(self.config_dir, config_file)
      if not os.path.exists(full_config_file):
        logging.warning(f"Config file: {full_config_file} do not exists!")
        return False
    return True

  def build(self):
    """Build the dataset."""
    if self.is_ready():
      logging.info("Dataset is ready.")
      return
    logging.info('Dataset is not ready.')
    if not os.path.exists(self.project_dir):
      os.mkdir(self.project_dir)
    if not os.path.exists(self.data_dir):
      os.mkdir(self.data_dir)
    if not os.path.exists(self.config_dir):
      os.mkdir(self.config_dir)
    self.copy_config_files()
    if not self._download_ready():
      self.download()
    self.after_download()
    logging.info("Dataset is ready.")
