/* Copyright (C) 2017 Beijing Didi Infinity Technology and Development Co.,Ltd.
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "kernels/delta_delta.h"

#include <math.h>

namespace delta {

const int kOrder = 2;
const int kWindow = 2;

DeltaDelta::DeltaDelta()
    : initialized_(false), order_(kOrder), window_(kWindow) {}

DeltaDelta::~DeltaDelta() {
    for (int i = 0; i < scales_.size(); i++)
      std::vector<double>().swap(scales_[i]);
    (scales_).clear();
}

bool DeltaDelta::Initialize(int order, int window) {
  if (order < 0 || order > 1000) {
    LOG(ERROR) << "order must be greater than zero and less than 1000.";
    return false;
  }
  if (window < 0 || window > 1000) {
    LOG(ERROR) << "window must be greater than zero and less than 1000.";
    return false;
  }
  order_ = order;
  window_ = window;

  scales_.resize(order_ + 1);
  scales_[0].resize(1);
  scales_[0][0] =
      1.0;  // trival window for 0th order delta [i.e. baseline feats]

  for (int i = 1; i <= order_; i++) {
    std::vector<double>&prev_scales = scales_[i - 1], &cur_scales = scales_[i];

    int window = window_;
    if (window == 0) {
      LOG(ERROR) << "window must not be zero.";
      return false;
    }
    int prev_offset = (static_cast<int>(prev_scales.size() - 1)) / 2,
        cur_offset = prev_offset + window;
    cur_scales.resize(prev_scales.size() + 2 * window);

    double normalizer = 0.0;
    for (int j = -window; j <= window; j++) {
      normalizer += j * j;
      for (int k = -prev_offset; k <= prev_offset; k++) {
        cur_scales[j + k + cur_offset] +=
            static_cast<double>(j) * prev_scales[k + prev_offset];
      }
    }

    for (int i = 0; i < cur_scales.size(); i++) {
      cur_scales[i] *= (1.0 / normalizer);
    }
  }

  initialized_ = true;
  return initialized_;
}

// process one frame per time
void DeltaDelta::Compute(const Tensor& input_feats, int frame,
                         std::vector<double>* output) const {
  if (!initialized_) {
    LOG(ERROR) << "DeltaDelta not initialized.";
    return;
  }

  int num_frames = input_feats.dim_size(0);
  int feat_dim = input_feats.dim_size(1);
  int output_dim = feat_dim * (order_ + 1);

  output->resize(output_dim);
  auto input = input_feats.matrix<float>();

  for (int i = 0; i <= order_; i++) {
    const std::vector<double>& scales = scales_[i];
    int max_offset = (scales.size() - 1) / 2;
    // e.g. max_offset=2, (-2, 2)
    for (int j = -max_offset; j <= max_offset; j++) {
      // if asked to read
      int offset_frame = frame + j;
      if (offset_frame < 0)
        offset_frame = 0;
      else if (offset_frame >= num_frames)
        offset_frame = num_frames - 1;

      // sacles[0] for `-max_offset` frame
      double scale = scales[j + max_offset];
      if (scale != 0.0) {
        for (int k = 0; k < feat_dim; k++) {
          (*output)[i + k * (order_ + 1)] += input(offset_frame, k) * scale;
        }
      }
    }
  }
  return;
}

void DeltaDelta::Compute_frame(const std::vector<std::vector<float>> &input_feats, int frame,
                std::vector<float>* output) const {

   if (!initialized_) {
   LOG(ERROR) << "DeltaDelta not initialized.";
   return;
   }

  int num_frames = input_feats.size();
  int feat_dim = input_feats[0].size();
  int output_dim = feat_dim * (order_ + 1);

  output->resize(output_dim);

  for (int i = 0; i <= order_; i++) {
    const std::vector<double>& scales = scales_[i];
    int max_offset = (scales.size() - 1) / 2;
    // e.g. max_offset=2, (-2, 2)
    for (int j = -max_offset; j <= max_offset; j++) {
      // if asked to read
      int offset_frame = frame + j;
      if (offset_frame < 0)
        offset_frame = 0;
      else if (offset_frame >= num_frames)
        offset_frame = num_frames - 1;

      // sacles[0] for `-max_offset` frame
      double scale = scales[j + max_offset];
      if (scale != 0.0) {
        for (int k = 0; k < feat_dim; k++) {
          (*output)[i + k * (order_ + 1)] += input_feats[offset_frame][k] * scale;
        }
      }
    }
  }
  return;
}

}  // namespace delta
