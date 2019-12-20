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
  i_snip_edges = true;
  i_remove_dc_offset = true;
  pf_FrmEng = NULL;
}

FramePow::~FramePow() { free(pf_FrmEng); }

void FramePow::set_window_length_sec(float window_length_sec) {
  window_length_sec_ = window_length_sec;
}

void FramePow::set_frame_length_sec(float frame_length_sec) {
  frame_length_sec_ = frame_length_sec;
}

void FramePow::set_snip_edges(bool snip_edges) { i_snip_edges = snip_edges; }

void FramePow::set_remove_dc_offset(bool remove_dc_offset) {
  i_remove_dc_offset = remove_dc_offset;
 }

int FramePow::init_eng(int input_size, float sample_rate) {
  f_SamRat = sample_rate;
  i_WinLen = static_cast<int>(window_length_sec_ * f_SamRat);
  i_FrmLen = static_cast<int>(frame_length_sec_ * f_SamRat);
  if (i_snip_edges == true)
    i_NumFrm = (input_size - i_WinLen) / i_FrmLen + 1;
  else
    i_NumFrm = (input_size + i_FrmLen / 2) / i_FrmLen;

  pf_FrmEng = static_cast<float*>(malloc(sizeof(float) * i_NumFrm));

  return 1;
}

int FramePow::proc_eng(const float* mic_buf, int input_size) {
  int i, n, k;
  float* win = static_cast<float*>(malloc(sizeof(float) * i_WinLen));

  for (n = 0; n < i_NumFrm; n++) {
    pf_FrmEng[n] = 0.0;
    float sum = 0.0;
    float energy = 0.0;
    for (k = 0; k < i_WinLen; k++) {
      int index = n * i_FrmLen + k;
      if (index < input_size)
        win[k] = mic_buf[index];
      else
        win[k] = 0.0f;
      sum += win[k];
    }

    if (i_remove_dc_offset == true) {
      float mean = sum / i_WinLen;
      for (int l = 0; l < i_WinLen; l++) win[l] -= mean;
    }

    for (i = 0; i < i_WinLen; i++) {
        energy += win[i] * win[i];
    }

    pf_FrmEng[n] = log(energy);

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
