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

#ifndef DELTA_LAYERS_OPS_KERNELS_CEPSTRUM_H_
#define DELTA_LAYERS_OPS_KERNELS_CEPSTRUM_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"

#include "complex_defines.h"
#include "spectrum.h"

namespace delta {
class Cepstrum {
 private:
  float window_length_sec_;
  float frame_length_sec_;

  float f_SamRat;
  int i_WinLen;
  int i_FrmLen;
  int i_NumFrm;
  int i_NumFrq;
  int i_FFTSiz;
  int i_NumCep;
  char s_FrqTyp[40];
  float f_LftExp;
  bool b_CMN;

  float* pf_spc;
  float* pf_wts;
  float* pf_dctm;
  float* pf_CEP;
  Spectrum* pclass_spc;

 public:
  Cepstrum();

  ~Cepstrum();

  void set_window_length_sec(float window_length_sec);

  void set_frame_length_sec(float frame_length_Sec);

  void set_ceps_subband_num(int ceps_subband_num);

  void set_adjacent_frame_num(int adjacent_frame_num);

  void set_do_ceps_mean_norm(bool do_ceps_mean_norm);

  int init_cep(int input_size, float sample_rate);

  int proc_cep(const float* mic_buf, int input_size);

  int get_cep(float* output);

  int write_cep();

  TF_DISALLOW_COPY_AND_ASSIGN(Cepstrum);
};
}  // namespace delta
#endif  // DELTA_LAYERS_OPS_KERNELS_CEPSTRUM_H_
