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
  i_snip_edges = 1;
  i_raw_energy = 1;
  f_PreEph = 0.97;
  i_remove_dc_offset = true;
  snprintf(s_WinTyp, sizeof(s_WinTyp), "povey");
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

void Spectrum::set_snip_edges(int snip_edges) { i_snip_edges = snip_edges; }

void Spectrum::set_raw_energy(int raw_energy) {i_raw_energy = raw_energy;}

void Spectrum::set_remove_dc_offset(bool remove_dc_offset) {i_remove_dc_offset = remove_dc_offset;}

void Spectrum::set_preEph(float preEph) {f_PreEph = preEph;}

void Spectrum::set_window_type(char* window_type){
    snprintf(s_WinTyp, sizeof(s_WinTyp), window_type);
}

int Spectrum::init_spc(int input_size, float sample_rate) {
  f_SamRat = sample_rate;
  i_WinLen = static_cast<int>(window_length_sec_ * f_SamRat);
  i_FrmLen = static_cast<int>(frame_length_sec_ * f_SamRat);
  if (i_snip_edges == 1)
    i_NumFrm = (input_size - i_WinLen) / i_FrmLen + 1;
  else
    i_NumFrm = (input_size + i_FrmLen / 2) / i_FrmLen;
//  snprintf(s_WinTyp, sizeof(s_WinTyp), "hamm");
  i_FFTSiz = static_cast<int>(pow(2.0f, ceil(log2(i_WinLen))));
  i_NumFrq = i_FFTSiz / 2 + 1;

  pf_WINDOW = static_cast<float*>(malloc(sizeof(float) * i_WinLen));
  pf_SPC = static_cast<float*>(malloc(sizeof(float) * i_NumFrq * i_NumFrm));

  return 1;
}

int Spectrum::proc_spc(const float* mic_buf, int input_size) {
  int n, k;
  bool switch_fbank = true;

  /* generate window */
  gen_window(pf_WINDOW, i_WinLen, s_WinTyp);
  do_preemphasis2();

  float tmp;
  xcomplex* win = static_cast<xcomplex*>(malloc(sizeof(xcomplex) * i_FFTSiz));
  float* win_buf = static_cast<float*>(malloc(sizeof(float) * i_WinLen));
  float* eph_buf = static_cast<float*>(malloc(sizeof(float) * i_WinLen));
  float* win_temp = static_cast<float*>(malloc(sizeof(float) * i_WinLen));
  xcomplex* fftwin =
      static_cast<xcomplex*>(malloc(sizeof(xcomplex) * i_FFTSiz));

  for (n = 0; n < i_NumFrm; n++) {
    float signal_raw_log_energy = 0.0;
    float sum = 0.0;
    for (int l = 0; l < i_WinLen; l++){
        int index = n * i_FrmLen + l;
        if (index < input_size)
            win_buf[l] = mic_buf[index];
        else
            win_buf[l] = 0.0f;
        sum += win_buf[l];
    }

    if (i_remove_dc_offset == true){
        float mean = sum / i_WinLen;
        for (int l = 0; l < i_WinLen; l++)
            win_buf[l] -= mean;
    }

    /* do pre-emphais */
    do_frame_preemphasis(win_buf, eph_buf, i_WinLen, f_PreEph);


    for (k = 0; k < i_WinLen; k++) {
      win[k].r = eph_buf[k] * pf_WINDOW[k];
      win[k].i = 0.0f;
      if (i_raw_energy == 1)
        win_temp[k] = win_buf[k];
      else
        win_temp[k] = win[k].r;
    }

    for (k = i_WinLen; k < i_FFTSiz; k++) {
      win[k].r = 0.0f;
      win[k].i = 0.0f;
    }

    /* raw energy */
    signal_raw_log_energy = compute_energy(win_temp, i_WinLen);

    /* fft */
    dit_r2_fft(win, fftwin, i_FFTSiz, -1);

    for (k = 0; k < i_NumFrq; k++) {
      if (k == 0 && switch_fbank == false){
        fftwin[k].r = sqrt(signal_raw_log_energy);
        fftwin[k].i = 0.0f;
      }
      if (i_OutTyp == 1)
        pf_SPC[n * i_NumFrq + k] = complex_abs2(fftwin[k]);
      else if (i_OutTyp == 2)
        pf_SPC[n * i_NumFrq + k] = log(complex_abs2(fftwin[k]));
      else
        return -1;
    }
  }

  free(win_temp);
  free(win_buf);
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
