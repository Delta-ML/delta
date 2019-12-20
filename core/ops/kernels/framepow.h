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

#ifndef DELTA_LAYERS_OPS_KERNELS_FRAMEPOW_H_
#define DELTA_LAYERS_OPS_KERNELS_FRAMEPOW_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"

using namespace tensorflow;  // NOLINT

namespace delta {
class FramePow {
 private:
  float window_length_sec_;
  float frame_length_sec_;
  bool i_snip_edges;
  bool i_remove_dc_offset;

  float f_SamRat;
  int i_WinLen;
  int i_FrmLen;
  int i_NumFrm;

  float* pf_FrmEng;

 public:
  FramePow();

  ~FramePow();

  void set_window_length_sec(float window_length_sec);

  void set_frame_length_sec(float frame_length_sec);

  void set_snip_edges(bool snip_edges);

  void set_remove_dc_offset(bool remove_dc_offset);

  int init_eng(int input_size, float sample_rate);

  int proc_eng(const float* mic_buf, int input_size);

  int get_eng(float* output);

  int write_eng();

  TF_DISALLOW_COPY_AND_ASSIGN(FramePow);
};
}  // namespace delta
#endif  // DELTA_LAYERS_OPS_KERNELS_FRAMEPOW_H_
