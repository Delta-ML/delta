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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "delta/layers/ops/kernels/pitch.h"
#include "delta/layers/ops/kernels/support_functions.h"

namespace delta {
const float window_length_sec = 0.025;
const float frame_length_sec = 0.010;
const float thres_autoc = 0.3;

Pitch::Pitch() {
  window_length_sec_ = window_length_sec;
  frame_length_sec_ = frame_length_sec;
  thres_autoc_ = thres_autoc;
  pf_PitEst = NULL;
}

Pitch::~Pitch() { free(pf_PitEst); }

void Pitch::set_window_length_sec(float window_length_sec) {
  window_length_sec_ = window_length_sec;
}

void Pitch::set_frame_length_sec(float frame_length_sec) {
  frame_length_sec_ = frame_length_sec;
}

void Pitch::set_thres_autoc(float thres_autoc) { thres_autoc_ = thres_autoc; }

int Pitch::init_pit(int input_size, float sample_rate) {
  f_SamRat = sample_rate;
  i_WinLen = static_cast<int>(window_length_sec_ * f_SamRat);
  i_FrmLen = static_cast<int>(frame_length_sec_ * f_SamRat);
  i_NumFrm = (input_size - i_WinLen) / i_FrmLen + 1;

  f_peaka = thres_autoc_;  // threshold in autocorr
  pf_PitEst = static_cast<float*>(calloc(i_NumFrm, sizeof(float)));

  return 1;
}

int Pitch::proc_pit(const float* mic_buf) {
  int n, m, k;
  float* pit_tmp = static_cast<float*>(calloc(i_NumFrm, sizeof(float)));
  float* win = static_cast<float*>(malloc(sizeof(float) * i_WinLen));
  int L = 300;
  L = static_cast<int>(fmin(L, i_WinLen));
  float* autoc = static_cast<float*>(malloc(sizeof(float) * (L + 1)));
  int flo = static_cast<int>(f_SamRat / 450.0f);
  int fhi = static_cast<int>(f_SamRat / 60.0f);
  float peaka;

  for (n = 0; n < i_NumFrm; n++) {
    for (k = 0; k < i_WinLen; k++) {
      win[k] = mic_buf[n * i_FrmLen + k];
    }
    compute_autoc(autoc, win, i_WinLen, L);
    peaka = f_peaka;
    for (m = flo; m < fhi; m++) {
      if ((autoc[m] >= autoc[m - 1]) && (autoc[m] >= autoc[m + 1]) &&
          (autoc[m] > peaka)) {
        peaka = autoc[m];
        pit_tmp[n] = f_SamRat / m;
      }
    }
  }
  // smooth output
  int idx_l, idx_h;
  float n_max, n_min;
  for (n = 0; n < i_NumFrm; n++) {
    idx_l = static_cast<int>(fmax(0.0f, static_cast<float>(n - 1)));
    idx_h = static_cast<int>(
        fmin(static_cast<float>(i_NumFrm - 1), static_cast<float>(n + 1)));
    n_max = fmax(pit_tmp[idx_l], pit_tmp[idx_h]);
    n_min = fmin(pit_tmp[idx_l], pit_tmp[idx_h]);
    if (pit_tmp[n] > n_max * 1.5 | pit_tmp[n] < n_min / 1.5) {
      pf_PitEst[n] = (pit_tmp[idx_l] + pit_tmp[idx_h]) / 2;
    } else {
      pf_PitEst[n] = pit_tmp[n];
    }
  }

  free(win);
  free(autoc);
  return 1;
}

int Pitch::get_pit(float* output) {
  memcpy(output, pf_PitEst, sizeof(float) * i_NumFrm);

  return 1;
}

int Pitch::write_pit() {
  FILE* fp;
  fp = fopen("pitch.txt", "w");
  int n;
  for (n = 0; n < i_NumFrm; n++) {
    fprintf(fp, "%4.6f\n", pf_PitEst[n]);
  }
  fclose(fp);
  return 1;
}

}  // namespace delta
