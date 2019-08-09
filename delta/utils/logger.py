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
''' logger utils'''

import os
from absl import flags
from absl import logging
import logging as _logging

FLAGS = flags.FLAGS


def set_logging(is_debug, config):
  absl_logger = logging.get_absl_logger()
  # create formatter and add it to the handlers
  formatter = _logging.Formatter("[ %(asctime)-15s %(levelname)s %(filename)15s:%(lineno)-4d " \
              " %(process)-5d ]  %(message)s")

  log_dir = config["solver"]["saver"]["model_path"]
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  logging.get_absl_handler().use_absl_log_file(
      program_name='delta', log_dir=log_dir)

  fh = _logging.StreamHandler()
  fh.setLevel(_logging.NOTSET)
  fh.setFormatter(formatter)
  absl_logger.addHandler(fh)

  if is_debug:
    logging.set_verbosity(_logging.DEBUG)
  else:
    logging.set_verbosity(_logging.NOTSET)

  logging.info("Also save log file to directory: {}".format(log_dir))
