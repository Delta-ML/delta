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

#include "kernels/framepow.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace delta {
const float window_length_sec = 0.025;
const float frame_length_sec = 0.010;

FramePow::FramePow() {
  window_length_sec_ = window_length_sec;
  frame_length_sec_ = frame_length_sec;
  pf_FrmEng = NULL;
}

FramePow::~FramePow() { free(pf_FrmEng); }

void FramePow::set_window_length_sec(float window_length_sec) {
  window_length_sec_ = window_length_sec;
}

void FramePow::set_frame_length_sec(float frame_length_sec) {
  frame_length_sec_ = frame_length_sec;
}

int FramePow::init_eng(int input_size, float sample_rate) {
  f_SamRat = sample_rate;
  i_WinLen = static_cast<int>(window_length_sec_ * f_SamRat);
  i_FrmLen = static_cast<int>(frame_length_sec_ * f_SamRat);
  i_NumFrm = (input_size - i_WinLen) / i_FrmLen + 1;

  pf_FrmEng = static_cast<float*>(malloc(sizeof(float) * i_NumFrm));

  return 1;
}

int FramePow::proc_eng(const float* mic_buf) {
  int n, k;
  float* win = static_cast<float*>(malloc(sizeof(float) * i_WinLen));

  for (n = 0; n < i_NumFrm; n++) {
    pf_FrmEng[n] = 0.0;
    for (k = 0; k < i_WinLen; k++) {
      win[k] = mic_buf[n * i_FrmLen + k];
      pf_FrmEng[n] = pf_FrmEng[n] + win[k] * win[k];
    }
  }

  free(win);
  return 1;
}

int FramePow::get_eng(float* output) {
  memcpy(output, pf_FrmEng, sizeof(float) * i_NumFrm);

  return 1;
}

int FramePow::write_eng() {
  FILE* fp;
  fp = fopen("frame_energy.txt", "w");
  int n;
  for (n = 0; n < i_NumFrm; n++) {
    fprintf(fp, "%4.6f\n", pf_FrmEng[n]);
  }
  fclose(fp);
  return 1;
}
}  // namespace delta
