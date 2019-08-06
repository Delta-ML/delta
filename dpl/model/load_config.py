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

import yaml
import sys
import os
from absl import logging


usage = f"python {sys.argv[0]} config_file"

if len(sys.argv) != 2:
  logging.error(usage)
  sys.exit(-1)

config_file = sys.argv[1]

if not os.path.exists(config_file):
  print(f"{config_file} not exist!")
  sys.exit(-1)

with open(config_file) as f:
  config = yaml.load(f, Loader=yaml.SafeLoader)

