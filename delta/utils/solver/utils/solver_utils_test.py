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
''' Test for solver_utils.py '''

import time
from pathlib import Path
import delta.compat as tf
from absl import logging

from delta import utils
from delta.utils.solver.utils import solver_utils

# pylint: disable=missing-docstring


class SolverUtilsTest(tf.test.TestCase):
  ''' Unit test for solver_utils. '''

  def setUp(self):
    super().setUp()
    self.model_path = self.get_temp_dir()
    self.file_name_pattern = 'model.{epoch:02d}-{monitor:02f}.ckpt'
    self.specified_model_file_name = 'model.09-1.00.ckpt'

  # pylint: disable=too-many-locals
  def test_get_model_file(self):
    ''' get model file according the specified model_load_type unittest'''
    model_load_dict = {
        'best':
            Path(self.model_path).joinpath('best_model.ckpt'),
        'specific':
            Path(self.model_path).joinpath(self.specified_model_file_name),
        'latest':
            Path(self.model_path).joinpath('model.00-1.00.ckpt')
    }

    # There is no model file in model_path
    mode_list = [utils.EVAL, utils.INFER]
    for cur_mode in mode_list:
      model_load_type_list = ["specific", "latest"]
      for cur_model_load_type in model_load_type_list:
        with self.assertRaises(AssertionError) as assert_err:
          _, _ = solver_utils.get_model_file(
              dir_name=self.model_path,
              file_name_pattern=self.file_name_pattern,
              mode=cur_mode,
              model_load_type=cur_model_load_type,
              specified_model_file_name=self.specified_model_file_name)
        the_exception = assert_err.exception
        self.assertEqual(
            str(the_exception),
            '{} END, since no model file is found'.format(cur_mode))

      model_load_type_list = [None, 'scratch', 'best']
      for cur_model_load_type in model_load_type_list:
        model_load_type, model_file_name = solver_utils.get_model_file(
            dir_name=self.model_path,
            file_name_pattern=self.file_name_pattern,
            mode=cur_mode,
            model_load_type=cur_model_load_type,
            specified_model_file_name=self.specified_model_file_name)
        self.assertEqual('best', model_load_type)
        self.assertEqual(model_load_dict['best'], model_file_name)

    cur_mode = utils.TRAIN
    model_load_type_list = [None, 'scratch', 'specific', "latest"]
    for cur_model_load_type in model_load_type_list:
      model_load_type, model_file_name = solver_utils.get_model_file(
          dir_name=self.model_path,
          file_name_pattern=self.file_name_pattern,
          mode=cur_mode,
          model_load_type=cur_model_load_type,
          specified_model_file_name=self.specified_model_file_name)
      self.assertEqual(model_load_type, 'scratch')
      self.assertIsNone(model_file_name)

    cur_model_load_type = 'best'
    model_load_type, model_file_name = solver_utils.get_model_file(
        dir_name=self.model_path,
        file_name_pattern=self.file_name_pattern,
        mode=cur_mode,
        model_load_type=cur_model_load_type,
        specified_model_file_name=self.specified_model_file_name)
    self.assertEqual(model_load_type, cur_model_load_type)
    self.assertEqual(model_file_name, model_load_dict[cur_model_load_type])
    # create model files in model_path
    file_name_list = [
        'best_model.ckpt', 'model.09-1.00.ckpt', 'model.00-1.00.ckpt'
    ]
    for file_name in file_name_list:
      file_path = Path(self.model_path).joinpath(file_name)
      with open(file_path, 'w', encoding='utf-8') as f:  #pylint: disable=invalid-name
        f.write('test ckpt file')
        time.sleep(1)

    all_mode_list = [utils.TRAIN, utils.EVAL, utils.INFER]
    for cur_mode in all_mode_list:
      for cur_model_load_type in model_load_dict:
        model_load_type, model_file_name = solver_utils.get_model_file(
            dir_name=self.model_path,
            file_name_pattern=self.file_name_pattern,
            mode=cur_mode,
            model_load_type=cur_model_load_type,
            specified_model_file_name=self.specified_model_file_name)
        self.assertEqual(cur_model_load_type, model_load_type)
        self.assertEqual(model_load_dict[cur_model_load_type], model_file_name)

    model_load_type_list = ['scratch', None]
    mode_list = [utils.EVAL, utils.INFER]
    for cur_model_load_type in model_load_type_list:
      for cur_mode in mode_list:
        model_load_type, model_file_name = solver_utils.get_model_file(
            dir_name=self.model_path,
            file_name_pattern=self.file_name_pattern,
            mode=cur_mode,
            model_load_type=cur_model_load_type,
            specified_model_file_name=self.specified_model_file_name)
        self.assertEqual('best', model_load_type)
        self.assertEqual(
            Path(self.model_path).joinpath('best_model.ckpt'), model_file_name)

    cur_mode = utils.TRAIN
    cur_model_load_type = 'scratch'
    model_load_type, model_file_name = solver_utils.get_model_file(
        dir_name=self.model_path,
        file_name_pattern=self.file_name_pattern,
        mode=cur_mode,
        model_load_type=cur_model_load_type,
        specified_model_file_name=self.specified_model_file_name)
    self.assertEqual(cur_model_load_type, model_load_type)
    self.assertIsNone(model_file_name)

    cur_model_load_type = None
    model_load_type, model_file_name = solver_utils.get_model_file(
        dir_name=self.model_path,
        file_name_pattern=self.file_name_pattern,
        mode=cur_mode,
        model_load_type=cur_model_load_type,
        specified_model_file_name=self.specified_model_file_name)
    self.assertEqual('latest', model_load_type)
    self.assertEqual(
        Path(self.model_path).joinpath('model.00-1.00.ckpt'), model_file_name)

  def test_get_most_recently_modified_file_matching_pattern(self):
    ''' get the most recently modified model file matching pattern unittest'''
    # There is no model file in model_path
    most_rencently_modified_file = solver_utils.get_most_recently_modified_file_matching_pattern(
        self.model_path, self.file_name_pattern)
    self.assertIsNone(most_rencently_modified_file)

    #pylint: disable=invalid-name
    with open(
        Path(self.model_path).joinpath('model.01-1.00.h5'),
        'w',
        encoding='utf-8') as f:
      f.write('test ckpt file')
    most_rencently_modified_file = solver_utils.get_most_recently_modified_file_matching_pattern(
        self.model_path, self.file_name_pattern)
    self.assertIsNone(most_rencently_modified_file)

    file_name_list = [
        'model.01-1.00.ckpt', 'model.02-1.00.ckpt', 'model.03-1.00.ckpt'
    ]
    for file_name in file_name_list:
      file_path = Path(self.model_path).joinpath(file_name)
      with open(file_path, 'w', encoding='utf-8') as f:  #pylint: disable=invalid-name
        f.write('test ckpt file')
        time.sleep(1)
    most_rencently_modified_file = solver_utils.get_most_recently_modified_file_matching_pattern(
        self.model_path, self.file_name_pattern)
    self.assertEqual(most_rencently_modified_file.name, file_name_list[-1])


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  tf.enable_eager_execution()
  tf.test.main()
