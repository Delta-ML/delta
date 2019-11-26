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
''' HTK Reader'''
import os
import math
import numpy as np


class HtkReaderIO:
  """
    read the kaldi ark format file
    """

  def __init__(self):
    self._fp = None
    self._open_file_name = ""

    self._delta_normalizer = 0
    self._delta_window = 1

    self._calutlate_mean_variance_start = True
    self._meam_variance_list = []
    self._mean_variance_statu = False
    self._mean_array = None
    self._variance_array = None

  def _compute_delta_normalizer(self, delta_window=2):
    """
        compute the delta normalizer
        args:
            delta_window: the detla window
        return:
            -1: failed
            0: success
        """
    self._delta_normalizer = 0
    self._delta_window = delta_window

    self._delta_window_array = np.array(
        range(-1 * self._delta_window, self._delta_window + 1))
    self._delta_normalizer = np.sum(np.square(self._delta_window_array))
    self._delta_normalizer = 1.0 / self._delta_normalizer
    self._delta_window_array = self._delta_window_array * self._delta_normalizer

    # base method
    #for tmp_window in range(-1*delta_window, delta_window+1):
    #    self._delta_normalizer += tmp_window * tmp_window

    #self._delta_normalizer = 1.0 / self._delta_normalizer
    return 0

  def add_delta(self, feat_array, delta_order=2, delta_window=2):
    """
        add delta to feat
        args:
            feat_array: input feat which is from one sentence
            delta_order: the order of delta
            delta_window: the window of delta
        return:
            -1, array: failed
            0, feat_array: success
        """
    if self._delta_normalizer == 0 or self._delta_window != delta_window:
      self._compute_delta_normalizer(delta_window)

    frame_num, feat_dim = feat_array.shape
    tmp_output_delta_feat = [feat_array]
    for _ in range(1, delta_order + 1):
      tmp_delta_feat = np.zeros([frame_num, feat_dim])
      for tmp_frame_num in range(frame_num):
        tmp_action_frame_begin = tmp_frame_num - delta_window
        tmp_action_frame_end = tmp_frame_num + delta_window
        if tmp_action_frame_begin < 0 or tmp_action_frame_end >= frame_num:
          for tmp_detal_window in range(-1 * delta_window, delta_window + 1):
            tmp_action_frame = tmp_frame_num + tmp_detal_window
            if tmp_action_frame < 0:
              tmp_action_frame = 0
              fill_status = True
            if tmp_action_frame >= frame_num:
              tmp_action_frame = frame_num - 1
              fill_status = True

            tmp_delta_feat[tmp_frame_num] += (
                tmp_detal_window * tmp_output_delta_feat[-1][tmp_action_frame])

          tmp_delta_feat[tmp_frame_num] = (
              tmp_delta_feat[tmp_frame_num] * self._delta_normalizer)
        else:
          tmp_delta_feat[tmp_frame_num] = (
              np.sum(
                  (tmp_output_delta_feat[-1]
                   [tmp_action_frame_begin:tmp_action_frame_end + 1].T *
                   self._delta_window_array).T,
                  axis=0))
      tmp_output_delta_feat.append(tmp_delta_feat)
    output_delta_feat = np.hstack(tmp_output_delta_feat)
    return 0, output_delta_feat

  def calculate_mean_variance(self, feat_list, is_end=False):
    """
        calculate the mean and variance of the feat
        args:
            feat_list: feat list such as [[key, array]...]
            is_end: if is the last feat_list, set it to True,
                    else set it to False
        return:
            -1, []: failed
            0, []: when is_end is True, return the mean and variance list
                   else return empty list
        """
    if self._calutlate_mean_variance_start == True:
      self._meam_variance_list = [
          0,
          np.zeros([feat_list[0][1].shape[1]]),
          np.zeros([feat_list[0][1].shape[1]])
      ]
      self._calutlate_mean_variance_start = False

    for tmp_feat_list in feat_list:
      self._meam_variance_list[0] += tmp_feat_list[1].shape[0]
      self._meam_variance_list[1] += np.sum(tmp_feat_list[1], axis=0)
      self._meam_variance_list[2] += (
          np.sum(np.multiply(tmp_feat_list[1], tmp_feat_list[1]), axis=0))

    if is_end:
      self._calutlate_mean_variance_start = True
      final_mean = self._meam_variance_list[1] / self._meam_variance_list[0]
      final_variance = (
          self._meam_variance_list[2] / self._meam_variance_list[0] -
          np.multiply(final_mean, final_mean))
      output_list = [["mean", final_mean], ["variance", final_variance]]
      return 0, output_list
    return 0, []

  def _read_mean_variance(self, mean_variance_file, dim_number):
    """
        read mean variance file
        args:
            mean_variance_file: the input mean variance file
            dim_number: gen dim_number array
        return:
            -1: failed
            0: success
        """
    if not os.path.exists(mean_variance_file):
      return -1

    with open(mean_variance_file) as fp:
      input_list = fp.readlines()
      input_dim_length = len(input_list)
      if dim_number > input_dim_length:
        return -1

      self._mean_array = np.zeros([dim_number])
      self._variance_array = np.zeros([dim_number])
      for input_line_number in range(0, dim_number):
        tmp_mean, tmp_variance = input_list[input_line_number].strip().split()
        self._mean_array[input_line_number] = float(tmp_mean)
        self._variance_array[input_line_number] = math.sqrt(float(tmp_variance))
    self._mean_variance_statu = True
    return 0

  def normalization_feat_by_mean_variance(self, feat_array, mean_variance_file):
    """
        normalization feat using mean variance
        args:
            feat_array: input feat array
            mean_variance_file: input mean variance file
        return:
            -1, array: failed
            0, array: success and return the new feat array
        """
    if not self._mean_variance_statu:
      if self._read_mean_variance(mean_variance_file, feat_array.shape[1]) < 0:
        print("[ERROR] _read_mean_variance failed with shape %s %s" %
              (feat_array.shape[0], feat_array.shape[1]))
        return -1, np.array([])

    mean_array = np.array([self._mean_array] * feat_array.shape[0])
    variance_array = np.array([self._variance_array] * feat_array.shape[0])

    new_feat_array = (feat_array - mean_array) / variance_array
    return 0, new_feat_array

  def splice_frames(self, feat_array, left_context, right_context):
    """
        splice the feature
        args:
            feat_array: the input feature
            left_context: int type and must >= 0
            right_context: int type and must >= 0
        return:
            -1, array: failed
            0, array: success
        """
    if left_context < 0 or right_context < 0:
      return -1, np.array([])

    feat_row, feat_dim = feat_array.shape
    length_for_per_feat = 1 + left_context + right_context
    #aim_feat_dim = feat_dim * length_for_per_feat
    output_list = []
    for row_number in range(feat_row):
      read_row_number_begin = row_number - left_context
      read_row_number_end = row_number + right_context
      if read_row_number_begin < 0 or read_row_number_end >= feat_row:
        tmp_output_list = []
        for tmp_row_number in range(length_for_per_feat):
          read_row_number = row_number + tmp_row_number - left_context
          if read_row_number < 0:
            read_row_number = 0

          if read_row_number >= feat_row:
            read_row_number = feat_row - 1

          tmp_output_list.append(feat_array[read_row_number])
        tmp_output_array = np.hstack(tmp_output_list)
        output_list.append(tmp_output_array)
      else:
        output_list.append(
            feat_array[read_row_number_begin:read_row_number_end + 1].flatten())
    output_array = np.vstack(output_list)
    return 0, output_array
