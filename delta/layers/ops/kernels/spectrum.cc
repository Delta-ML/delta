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

#include "kernels/spectrum.h"
#include "kernels/support_functions.h"

namespace delta {
const float window_length_sec = 0.025;
const float frame_length_sec = 0.010;

Spectrum::Spectrum() {
  window_length_sec_ = window_length_sec;
  frame_length_sec_ = frame_length_sec;
  i_OutTyp = 1;
  pf_WINDOW = NULL;
  pf_SPC = NULL;
}

Spectrum::~Spectrum() {
  free(pf_WINDOW);
  free(pf_SPC);
}

void Spectrum::set_window_length_sec(float window_length_sec) {
  window_length_sec_ = window_length_sec;
}

void Spectrum::set_frame_length_sec(float frame_length_sec) {
  frame_length_sec_ = frame_length_sec;
}

void Spectrum::set_output_type(int output_type) { i_OutTyp = output_type; }

int Spectrum::init_spc(int input_size, float sample_rate) {
  f_SamRat = sample_rate;
  i_WinLen = static_cast<int>(window_length_sec_ * f_SamRat);
  i_FrmLen = static_cast<int>(frame_length_sec_ * f_SamRat);
  i_NumFrm = (input_size - i_WinLen) / i_FrmLen + 1;
  f_PreEph = 0.97;
  snprintf(s_WinTyp, sizeof(s_WinTyp), "hamm");
  i_FFTSiz = static_cast<int>(pow(2.0f, ceil(log2(i_WinLen))));
  i_NumFrq = i_FFTSiz / 2 + 1;

  pf_WINDOW = static_cast<float*>(malloc(sizeof(float) * i_WinLen));
  pf_SPC = static_cast<float*>(malloc(sizeof(float) * i_NumFrq * i_NumFrm));

  return 1;
}

int Spectrum::proc_spc(const float* mic_buf, int input_size) {
  int n, k;

  /* generate window */
  gen_window(pf_WINDOW, i_WinLen, s_WinTyp);

  /* do pre-emphais */
  float* eph_buf =
      static_cast<float*>(malloc(sizeof(float) * (input_size + 1)));
  do_preemphasis(f_PreEph, eph_buf, mic_buf, input_size);

  float tmp;
  xcomplex* win = static_cast<xcomplex*>(malloc(sizeof(xcomplex) * i_FFTSiz));
  xcomplex* fftwin =
      static_cast<xcomplex*>(malloc(sizeof(xcomplex) * i_FFTSiz));

  for (n = 0; n < i_NumFrm; n++) {
    for (k = 0; k < i_WinLen; k++) {
      tmp = eph_buf[n * i_FrmLen + k];
      win[k].r = tmp * pf_WINDOW[k];
      win[k].i = 0.0f;
    }
    for (k = i_WinLen; k < i_FFTSiz; k++) {
      win[k].r = 0.0f;
      win[k].i = 0.0f;
    }

    /* fft */
    dit_r2_fft(win, fftwin, i_FFTSiz, -1);

    for (k = 0; k < i_NumFrq; k++) {
      if (i_OutTyp == 1)
        pf_SPC[n * i_NumFrq + k] = complex_abs2(fftwin[k]);
      else if (i_OutTyp == 2)
        pf_SPC[n * i_NumFrq + k] = log(complex_abs2(fftwin[k]));
      else
        return -1;
    }
  }

  free(eph_buf);
  free(win);
  free(fftwin);

  return 1;
}

int Spectrum::get_spc(float* output) {
  int n, m;
  for (m = 0; m < i_NumFrq; m++) {
    for (n = 0; n < i_NumFrm; n++) {
      output[n * i_NumFrq + m] = pf_SPC[n * i_NumFrq + m];
    }
  }
  return 1;
}

int Spectrum::write_spc() {
  FILE* fp;
  fp = fopen("spectrum.txt", "w");
  int n, m;
  for (n = 0; n < i_NumFrm; n++) {
    for (m = 0; m < i_NumFrq; m++) {
      fprintf(fp, "%4.6f  ", pf_SPC[n * i_NumFrq + m]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
  return 1;
}
}  // namespace delta
