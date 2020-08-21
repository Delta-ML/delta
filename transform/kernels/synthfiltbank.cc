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

#include "support_functions.h"
#include "synthfiltbank.h"

namespace delta {
const float window_length_sec = 0.030;
const float frame_length_sec = 0.010;

Synthfiltbank::Synthfiltbank() {
  window_length_sec_ = window_length_sec;
  frame_length_sec_ = frame_length_sec;
  pf_WINDOW = NULL;
  pf_wav = NULL;
}

Synthfiltbank::~Synthfiltbank() {
  free(pf_WINDOW);
  free(pf_wav);
}

void Synthfiltbank::set_window_length_sec(float window_length_sec) {
  window_length_sec_ = window_length_sec;
}

void Synthfiltbank::set_frame_length_sec(float frame_length_sec) {
  frame_length_sec_ = frame_length_sec;
}

int Synthfiltbank::init_sfb(int num_frq, int num_frm, float sample_rate) {
  f_SamRat = sample_rate;
  i_WinLen = static_cast<int>(window_length_sec_ * f_SamRat);
  i_FrmLen = static_cast<int>(frame_length_sec_ * f_SamRat);
  i_NumFrm = num_frm;
  i_WavLen = (i_NumFrm - 1) * i_FrmLen + i_WinLen;
  snprintf(s_WinTyp, sizeof(s_WinTyp), "rect");
  i_NumFrq = num_frq;
  i_FFTSiz = (i_NumFrq - 1) * 2;

  pf_WINDOW = static_cast<float*>(malloc(sizeof(float) * i_WinLen));
  pf_wav = static_cast<float*>(calloc(i_WavLen, sizeof(float)));

  return 1;
}

int Synthfiltbank::proc_sfb(const float* powspc, const float* phaspc) {
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
    for (k = 0; k < i_NumFrq; k++) {
      tmp = sqrt(powspc[n * i_NumFrq + k]);
      fftwin[k].r = tmp * cos(phaspc[n * i_NumFrq + k]);
      fftwin[k].i = tmp * sin(phaspc[n * i_NumFrq + k]);
    }
    for (k = i_NumFrq; k < i_FFTSiz; k++) {
      fftwin[k].r = fftwin[i_FFTSiz - k].r;
      fftwin[k].i = -1.0f * fftwin[i_FFTSiz - k].i;
    }
    /* ifft */
    dit_r2_fft(fftwin, win, fft_buf, i_FFTSiz, 1);

    for (k = 0; k < i_WinLen; k++) {
      pf_wav[n * i_FrmLen + k] +=
          win[k].r / (i_WinLen / i_FrmLen) / pf_WINDOW[k];
    }
  }

  free(win);
  free(fftwin);
  free(fft_buf);
  return 1;
}

int Synthfiltbank::get_sfb(float* output) {
  int n;
  for (n = 0; n < i_WavLen; n++) {
    output[n] = pf_wav[n];
  }
  return 1;
}

int Synthfiltbank::write_sfb() {
  FILE* fp;
  fp = fopen("sfb_wav.pcm", "wb");
  if (fp == NULL) {
    printf("open output file {sfb_wav.pcm} error.\n");
    exit(-1);
  }
  int16 tmp;
  for (int n = 0; n < i_WavLen; n++) {
    tmp = static_cast<int16>(pf_wav[n] * pow(2, 15));
    fwrite(&tmp, sizeof(int16), 1, fp);
  }
  fclose(fp);

  FILE* fp1;
  fp1 = fopen("sfb_wav.txt", "wb");
  if (fp1 == NULL) {
    printf("open output file {sfb_wav.txt} error.\n");
    exit(-1);
  }
  for (int n = 0; n < i_WavLen; n++) {
    fprintf(fp1, "%4.12f\n", pf_wav[n]);
  }
  fclose(fp1);

  return 1;
}

}  // namespace delta
