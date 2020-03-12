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

from absl import logging
from delta.utils.register import registers


def build_dataset(dataset_name, dataset_dir):
  if dataset_name not in registers.dataset:
    logging.warning(f"Dataset: {dataset_name} not supported!")
  ds_cls = registers.dataset[dataset_name]
  ds_obj = ds_cls(dataset_dir)
  res = ds_obj.build()
  if not res:
    logging.info(f"Dataset: {dataset_name} built failed!")
    return
  logging.info(f"Dataset: {dataset_name} built successfully.")
