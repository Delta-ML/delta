#!/usr/bin/env python3
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

import argparse
import os
from absl import logging
import delta.compat as tf
from tensorflow.python.platform import gfile


def edit_pb_txt(old_args, export_dir):
  """
  Edit file path argument in pbtxt file.
  :param old_args: Old file paths need to be copied and edited.
  :param export_dir: Directory of the saved model.
  """
  assets_extra_dir = os.path.join(export_dir, "./assets.extra")
  if not os.path.exists(assets_extra_dir):
    os.makedirs(assets_extra_dir)

  new_args = []
  for one_old in old_args:
    if not os.path.exists(one_old):
      raise ValueError("{} do not exists!".format(one_old))
    one_new = os.path.join(assets_extra_dir, os.path.basename(one_old))
    new_args.append(one_new)
    logging.info("Copy file: {} to: {}".format(one_old, one_new))
    gfile.Copy(one_old, one_new, overwrite=True)

  pbtxt_file = os.path.join(export_dir, "saved_model.pbtxt")
  tmp_file = pbtxt_file + ".tmp"
  logging.info("Editing pbtxt file: {}".format(pbtxt_file))
  with open(pbtxt_file, "rt") as fin, open(tmp_file, "wt") as fout:
    for line in fin:
      for one_old, one_new in zip(old_args, new_args):
        line = line.replace(one_old, one_new)
      fout.write(line)
  gfile.Copy(tmp_file, pbtxt_file, overwrite=True)
  gfile.Remove(tmp_file)


if "__main__" in __name__:
  ap = argparse.ArgumentParser(
      description="Edit file path argument in pbtxt file.")
  ap.add_argument('--old_args', type=str, help="Old arguments, split by comma.")
  ap.add_argument(
      '--export_dir', type=str, help="Directory of the exported saved model.")
  args = ap.parse_args()
  logging.set_verbosity(logging.INFO)
  old_args = args.old_args.split(",")
  export_dir = args.export_dir
  edit_pb_txt(old_args, export_dir)
