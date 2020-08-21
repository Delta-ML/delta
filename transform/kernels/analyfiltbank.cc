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

#include "analyfiltbank.h"
#include "support_functions.h"

namespace delta {
const float window_length_sec = 0.030;
const float frame_length_sec = 0.010;

Analyfiltbank::Analyfiltbank() {
  window_length_sec_ = window_length_sec;
  frame_length_sec_ = frame_length_sec;
  pf_WINDOW = NULL;
  pf_PowSpc = NULL;
  pf_PhaSpc = NULL;
}

Analyfiltbank::~Analyfiltbank() {
  free(pf_WINDOW);
  free(pf_PowSpc);
  free(pf_PhaSpc);
}

void Analyfiltbank::set_window_length_sec(float window_length_sec) {
  if (fabs(window_length_sec - 0.030) > 0.00001) {
    printf("window_length {%f} has been set to 0.030 sec\n", window_length_sec);
    window_length_sec = 0.030;
  }
  window_length_sec_ = window_length_sec;
}

void Analyfiltbank::set_frame_length_sec(float frame_length_sec) {
  if (fabs(frame_length_sec - 0.010) > 0.00001) {
    printf("frame_length {%f} has been set to 0.010 sec\n", frame_length_sec);
    frame_length_sec = 0.010;
  }
  frame_length_sec_ = frame_length_sec;
}

int Analyfiltbank::init_afb(int input_size, float sample_rate) {
  i_WavLen = input_size;
  f_SamRat = sample_rate;
  i_WinLen = static_cast<int>(window_length_sec_ * f_SamRat);
  i_FrmLen = static_cast<int>(frame_length_sec_ * f_SamRat);
  i_NumFrm = (i_WavLen - i_WinLen) / i_FrmLen + 1;
  snprintf(s_WinTyp, sizeof(s_WinTyp), "rect");
  i_FFTSiz = static_cast<int>(pow(2.0f, ceil(log2(i_WinLen))));
  i_NumFrq = i_FFTSiz / 2 + 1;

  pf_WINDOW = static_cast<float*>(malloc(sizeof(float) * i_WinLen));
  pf_PowSpc = static_cast<float*>(malloc(sizeof(float) * i_NumFrq * i_NumFrm));
  pf_PhaSpc = static_cast<float*>(malloc(sizeof(float) * i_NumFrq * i_NumFrm));

  return 1;
}

int Analyfiltbank::proc_afb(const float* mic_buf) {
  int n, k;
  float tmp;
  xcomplex* win = static_cast<xcomplex*>(malloc(sizeof(xcomplex) * i_FFTSiz));
  xcomplex* fftwin =
      static_cast<xcomplex*>(malloc(sizeof(xcomplex) * i_FFTSiz));
  float* fft_buf = static_cast<float*>(malloc(sizeof(float) * 2 * i_FFTSiz));

  /* generate window */
  gen_window(pf_WINDOW, i_WinLen, s_WinTyp);

  /* frequency domain */
  for (n = 0; n < i_NumFrm; n++) {
    for (k = 0; k < i_WinLen; k++) {
      tmp = mic_buf[n * i_FrmLen + k];
      win[k].r = tmp * pf_WINDOW[k];
      win[k].i = 0.0f;
    }
    for (k = i_WinLen; k < i_FFTSiz; k++) {
      win[k].r = 0.0f;
      win[k].i = 0.0f;
    }

    /* fft */
    dit_r2_fft(win, fftwin, fft_buf, i_FFTSiz, -1);

    for (k = 0; k < i_NumFrq; k++) {
      pf_PowSpc[n * i_NumFrq + k] = complex_abs2(fftwin[k]);
      pf_PhaSpc[n * i_NumFrq + k] = atan2(fftwin[k].i, fftwin[k].r);
    }
  }

  free(win);
  free(fftwin);
  free(fft_buf);

  return 1;
}

int Analyfiltbank::get_afb(float* powspc, float* phaspc) {
  int n, m;
  for (m = 0; m < i_NumFrq; m++) {
    for (n = 0; n < i_NumFrm; n++) {
      powspc[n * i_NumFrq + m] = pf_PowSpc[n * i_NumFrq + m];
      phaspc[n * i_NumFrq + m] = pf_PhaSpc[n * i_NumFrq + m];
    }
  }
  return 1;
}

int Analyfiltbank::write_afb() {
  FILE* fp;
  fp = fopen("afb_pow_spectrum.txt", "w");
  int n, m;
  for (n = 0; n < i_NumFrm; n++) {
    for (m = 0; m < i_NumFrq; m++) {
      fprintf(fp, "%4.12f\n", pf_PowSpc[n * i_NumFrq + m]);
    }
  }
  fclose(fp);

  fp = fopen("afb_pha_spectrum.txt", "w");
  for (n = 0; n < i_NumFrm; n++) {
    for (m = 0; m < i_NumFrq; m++) {
      fprintf(fp, "%4.12f\n", pf_PhaSpc[n * i_NumFrq + m]);
    }
  }
  fclose(fp);
  return 1;
}

}  // namespace delta
