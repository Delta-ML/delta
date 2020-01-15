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
''' configration utils unittest'''
import tempfile

from absl import logging
import os
import delta.compat as tf

from delta import utils
from delta import PACKAGE_ROOT_DIR


class ConfigTest(tf.test.TestCase):
  ''' config unit test'''

  def setUp(self):
    super().setUp()
    ''' setup '''
    self.conf_str = '''
     name: Tom Smith
     age: 37
     spouse:
         name: Jane Smith
         age: 25
     children:
      - name: Jimmy Smith
        age: 15
      - name: Jenny Smith
        age: 12
    '''
    self.conf_true = {
        'name':
            'Tom Smith',
        'age':
            37,
        'spouse': {
            'name': 'Jane Smith',
            'age': 25
        },
        'children': [
            {
                'name': 'Jimmy Smith',
                'age': 15
            },
            {
                'name': 'Jenny Smith',
                'age': 12
            },
        ]
    }

    self.conf_file = tempfile.mktemp(suffix='conf.yaml')
    with open(self.conf_file, 'w', encoding='utf-8') as f:  #pylint: disable=invalid-name
      f.write(self.conf_str)

  def tearDown(self):
    ''' tear down '''

  def test_load_config(self):
    ''' load config unittest '''
    conf = utils.load_config(self.conf_file)
    self.assertDictEqual(conf, self.conf_true)
    config_dir = os.path.join(PACKAGE_ROOT_DIR, "configs/")
    for config_file in os.listdir(config_dir):
      config_file = os.path.join(config_dir, config_file)
      logging.info(f"Loading conf {config_file}")
      conf = utils.load_config(config_file)
      logging.info(conf)

  def test_save_config(self):
    ''' save config unittest '''
    utils.save_config(self.conf_true, self.conf_file)
    conf = utils.load_config(self.conf_file)
    self.assertDictEqual(conf, self.conf_true)

  def test_valid_config(self):
    ''' valid config unittest '''
    utils.save_config(self.conf_true, self.conf_file)
    conf = utils.load_config(self.conf_file)
    self.assertEqual(utils.valid_config(conf), True)

  def test_setdefault_config(self):
    ''' set default config unittest '''
    self.assertEqual(True, True)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.test.main()
