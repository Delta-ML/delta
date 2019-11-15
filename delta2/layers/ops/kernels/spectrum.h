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

#ifndef DELTA_LAYERS_OPS_KERNELS_SPECTRUM_H_
#define DELTA_LAYERS_OPS_KERNELS_SPECTRUM_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"

#include "kernels/complex_defines.h"

using namespace tensorflow;  // NOLINT

namespace delta {
class Spectrum {
 private:
  float window_length_sec_;
  float frame_length_sec_;

  float f_SamRat;
  int i_WinLen;
  int i_FrmLen;
  int i_NumFrm;
  int i_NumFrq;
  int i_FFTSiz;
  float f_PreEph;
  char s_WinTyp[40];
  int i_OutTyp;  // 1: PSD, 2:log(PSD)

  float* pf_WINDOW;
  float* pf_SPC;

 public:
  Spectrum();

  ~Spectrum();

  void set_window_length_sec(float window_length_sec);

  void set_frame_length_sec(float frame_length_sec);

  void set_output_type(int output_type);

  int init_spc(int input_size, float sample_rate);

  int proc_spc(const float* mic_buf, int input_size);

  int get_spc(float* output);

  int write_spc();

  TF_DISALLOW_COPY_AND_ASSIGN(Spectrum);
};
}  // namespace delta
#endif  // DELTA_LAYERS_OPS_KERNELS_SPECTRUM_H_
