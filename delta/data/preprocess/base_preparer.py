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
'''Base class for Preparer'''

import os
from pathlib import Path


class Preparer:
  '''Base class for Preparer'''

  def __init__(self, config):
    self.config = config
    self.reuse = self.config["data"]["task"]["preparer"].get("reuse", True)
    self.done_sign = self.config["data"]["task"]["preparer"].get("done_sign", "")

  def skip_prepare(self):
    """Check if task need to skip the prepare process."""
    return self.done_sign != "" and os.path.exists(self.done_sign) \
           and self.reuse

  def done_prepare(self):
    """Touch a sign file after the prepare process is done."""
    if self.done_sign != "" and not os.path.exists(self.done_sign):
      Path(self.done_sign).touch()

  def do_prepare(self, pre_process_pipeline):
    """Do the prepare processing."""
    raise NotImplementedError
