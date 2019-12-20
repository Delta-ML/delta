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
  i_snip_edges = true;
  i_raw_energy = 1;
  f_PreEph = 0.97;
  i_is_fbank = true;
  i_remove_dc_offset = true;
  i_dither = 0.0;
  snprintf(s_WinTyp, sizeof(s_WinTyp), "povey");
  pf_WINDOW = NULL;
  pf_SPC = NULL;
  win_temp = NULL;
  win_buf = NULL;
  eph_buf = NULL;
  win = NULL;
  fftwin = NULL;
  fft_buf = NULL;
}

void Spectrum::set_window_length_sec(float window_length_sec) {
  window_length_sec_ = window_length_sec;
}

void Spectrum::set_frame_length_sec(float frame_length_sec) {
  frame_length_sec_ = frame_length_sec;
}

void Spectrum::set_output_type(int output_type) { i_OutTyp = output_type; }

void Spectrum::set_snip_edges(bool snip_edges) { i_snip_edges = snip_edges; }

void Spectrum::set_raw_energy(int raw_energy) {i_raw_energy = raw_energy;}

void Spectrum::set_is_fbank(bool is_fbank) {i_is_fbank = is_fbank;}

void Spectrum::set_remove_dc_offset(bool remove_dc_offset) {i_remove_dc_offset = remove_dc_offset;}

void Spectrum::set_preEph(float preEph) {f_PreEph = preEph;}

void Spectrum::set_dither(float dither) {i_dither = dither;}

void Spectrum::set_window_type(char* window_type){
    snprintf(s_WinTyp, sizeof(s_WinTyp), "%s", window_type);
}

int Spectrum::init_spc(int input_size, float sample_rate) {
  f_SamRat = sample_rate;
  i_WinLen = static_cast<int>(window_length_sec_ * f_SamRat);
  i_FrmLen = static_cast<int>(frame_length_sec_ * f_SamRat);
  if (i_snip_edges == true)
    i_NumFrm = (input_size - i_WinLen) / i_FrmLen + 1;
  else
    i_NumFrm = (input_size + i_FrmLen / 2) / i_FrmLen;
  if (i_NumFrm < 1)
    i_NumFrm = 1;
  i_FFTSiz = static_cast<int>(pow(2.0f, ceil(log2(i_WinLen))));
  i_NumFrq = i_FFTSiz / 2 + 1;

  return 1;
}

int Spectrum::proc_spc(const float* mic_buf, int input_size) {
  int n, k;

  if (input_size < i_WinLen)
    std::cerr<<"Wraning: The length of input data is shorter than "<< window_length_sec_ << " s." <<std::endl;

  //malloc
  pf_WINDOW = static_cast<float*>(malloc(sizeof(float) * i_WinLen));
  pf_SPC = static_cast<float*>(malloc(sizeof(float) * i_NumFrq * i_NumFrm));
  win = static_cast<xcomplex*>(malloc(sizeof(xcomplex) * i_FFTSiz));
  win_buf = static_cast<float*>(malloc(sizeof(float) * i_WinLen));
  eph_buf = static_cast<float*>(malloc(sizeof(float) * i_WinLen));
  win_temp = static_cast<float*>(malloc(sizeof(float) * i_WinLen));
  fftwin = static_cast<xcomplex*>(malloc(sizeof(xcomplex) * i_FFTSiz));
  fft_buf = static_cast<float*>(malloc(sizeof(float) * 2 * i_FFTSiz));  // c.r&c.i

  /* generate window */
  gen_window(pf_WINDOW, i_WinLen, s_WinTyp);

  for (n = 0; n < i_NumFrm; n++) {
    float signal_raw_log_energy = 0.0;
    float sum = 0.0;
    for (int l = 0; l < i_WinLen; l++){
      int index = n * i_FrmLen + l;
      if (index < input_size) {
        win_buf[l] = mic_buf[index];
      } else {
        win_buf[l] = 0.0f;
      }
      sum += win_buf[l];
    }

    if(i_dither != 0.0) {
      do_dither(win_buf, i_WinLen, i_dither);
    }

    if (i_remove_dc_offset == true){
      float mean = sum / i_WinLen;
      for (int l = 0; l < i_WinLen; l++) {
        win_buf[l] -= mean;
      }
    }

    /* do pre-emphais */
    do_frame_preemphasis(win_buf, eph_buf, i_WinLen, f_PreEph);

    for (k = 0; k < i_WinLen; k++) {
      win[k].r = eph_buf[k] * pf_WINDOW[k];
      win[k].i = 0.0f;
    }

    if (i_raw_energy == 1) {
      std::memcpy(win_temp, win_buf, i_WinLen * sizeof(float));
    }
    else {
      for (k = 0; k < i_WinLen; k++) {
        win_temp[k] = win[k].r;
      }
    }

    std::memset((void*)&(win[i_WinLen]), 0, sizeof(float) * 2 * (i_FFTSiz - i_WinLen));;

    /* raw energy */
    signal_raw_log_energy = compute_energy(win_temp, i_WinLen);

    /* fft */
    dit_r2_fft(win, fftwin, fft_buf, i_FFTSiz, -1);

    if (!i_is_fbank) {
      fftwin[0].r = sqrt(signal_raw_log_energy);
      fftwin[0].i = 0.0f;
    }

    if (i_OutTyp == 1) {
      for (k = 0; k < i_NumFrq; k++) {
        pf_SPC[n * i_NumFrq + k] = complex_abs2(fftwin[k]);
      }
    } else if (i_OutTyp == 2) {
      for (k = 0; k < i_NumFrq; k++) {
        pf_SPC[n * i_NumFrq + k] = log(complex_abs2(fftwin[k]));
      }
    } else {
      return -1;
    }
  }

  free(pf_WINDOW);
  free(win_temp);
  free(win_buf);
  free(eph_buf);
  free(win);
  free(fftwin);
  free(fft_buf);

  return 1;
}

int Spectrum::get_spc(float* output) {
  std::memcpy((void*)output, (void*)pf_SPC, \
		i_NumFrq * i_NumFrm * sizeof(float));
  free(pf_SPC);
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
