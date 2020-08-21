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

#include "cepstrum.h"
#include "support_functions.h"

const float DELTAP = 0.000030518f;  // 2^(-16)

namespace delta {
const float window_length_sec = 0.025;
const float frame_length_sec = 0.010;
const int ceps_subband_num = 13;
const bool tag_ceps_mean_norm = true;

Cepstrum::Cepstrum() {
  window_length_sec_ = window_length_sec;
  frame_length_sec_ = frame_length_sec;
  i_NumCep = ceps_subband_num;
  b_CMN = tag_ceps_mean_norm;
  pf_spc = NULL;
  pf_wts = NULL;
  pf_dctm = NULL;
  pf_CEP = NULL;
  pclass_spc = NULL;
}

Cepstrum::~Cepstrum() {
  free(pf_spc);
  free(pf_wts);
  free(pf_dctm);
  free(pf_CEP);
  if (pclass_spc != NULL) {
    delete pclass_spc;
    pclass_spc = NULL;
  }
}

void Cepstrum::set_window_length_sec(float window_length_sec) {
  window_length_sec_ = window_length_sec;
}

void Cepstrum::set_frame_length_sec(float frame_length_sec) {
  frame_length_sec_ = frame_length_sec;
}

void Cepstrum::set_ceps_subband_num(int ceps_subband_num) {
  i_NumCep = ceps_subband_num;
}

void Cepstrum::set_do_ceps_mean_norm(bool tag_ceps_mean_norm) {
  b_CMN = tag_ceps_mean_norm;
}

int Cepstrum::init_cep(int input_size, float sample_rate) {
  f_SamRat = sample_rate;
  i_WinLen = static_cast<int>(window_length_sec_ * f_SamRat);
  i_FrmLen = static_cast<int>(frame_length_sec_ * f_SamRat);
  i_NumFrm = (input_size - i_WinLen) / i_FrmLen + 1;
  i_FFTSiz = static_cast<int>(pow(2.0f, ceil(log2(i_WinLen))));
  i_NumFrq = i_FFTSiz / 2 + 1;
  snprintf(s_FrqTyp, sizeof(s_FrqTyp), "mel");
  f_LftExp = 0.6f;

  pf_spc = static_cast<float*>(malloc(sizeof(float) * i_NumFrq * i_NumFrm));
  pf_wts = static_cast<float*>(malloc(sizeof(float) * i_NumCep * i_NumFrq));
  pf_dctm = static_cast<float*>(malloc(sizeof(float) * i_NumCep * i_NumCep));
  pf_CEP = static_cast<float*>(malloc(sizeof(float) * i_NumCep * i_NumFrm));
  pclass_spc = NULL;
  pclass_spc = new Spectrum();
  pclass_spc->init_spc(input_size, sample_rate);

  return 1;
}

int Cepstrum::proc_cep(const float* mic_buf, int input_size) {
  int n, m, k;
  float* ceps1 =
      static_cast<float*>(malloc(sizeof(float) * i_NumCep * i_NumFrm));
  float* ceps2 =
      static_cast<float*>(malloc(sizeof(float) * i_NumCep * i_NumFrm));
  float tmp;

  /* compute spectrum */
  pclass_spc->proc_spc(mic_buf, input_size);
  pclass_spc->get_spc(pf_spc);

  /* generate frequency scale warping */
  gen_warpweights(f_SamRat, i_NumCep, i_NumFrq, s_FrqTyp, i_FFTSiz, pf_wts);

  for (n = 0; n < i_NumFrm; n++) {
    for (k = 0; k < i_NumCep; k++) {
      tmp = 0.0f;
      for (m = 0; m < i_NumFrq; m++) {
        tmp = tmp + pf_wts[k * i_NumFrq + m] * sqrt(pf_spc[n * i_NumFrq + m]);
      }
      tmp = pow(tmp, 2);
      ceps1[n * i_NumCep + k] = fmax(DELTAP, tmp);
    }
  }

  /* generate DCT matrix */
  gen_dctmatrix(i_NumCep, pf_dctm);

  for (n = 0; n < i_NumFrm; n++) {
    for (k = 0; k < i_NumCep; k++) {
      tmp = 0.0f;
      for (m = 0; m < i_NumCep; m++) {
        tmp = tmp + pf_dctm[k * i_NumCep + m] * log(ceps1[n * i_NumCep + m]);
      }
      ceps2[n * i_NumCep + k] = tmp;
    }
  }

  /* lift cepstrum */
  lift_cepstrum(f_LftExp, i_NumCep, i_NumFrm, ceps2, ceps1);

  /* cepstrum mean normalization */
  if (b_CMN) {
    do_ceps_mean_norm(i_NumCep, i_NumFrm, ceps1, pf_CEP);
  } else {
    memcpy(pf_CEP, ceps1, sizeof(float) * i_NumCep * i_NumFrm);
  }

  free(ceps1);
  free(ceps2);
  return 1;
}

int Cepstrum::get_cep(float* output) {
  int n, m;
  for (n = 0; n < i_NumFrm; n++) {
    for (m = 0; m < i_NumCep; m++) {
      output[n * i_NumCep + m] = pf_CEP[n * i_NumCep + m];
    }
  }
  return 1;
}

int Cepstrum::write_cep() {
  FILE* fp;
  fp = fopen("cepstrum.txt", "w");
  int n, m;
  for (n = 0; n < i_NumFrm; n++) {
    for (m = 0; m < i_NumCep; m++) {
      fprintf(fp, "%4.6f  ", pf_CEP[n * i_NumCep + m]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
  return 1;
}
}  // namespace delta
