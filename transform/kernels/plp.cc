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

#include "plp.h"
#include "support_functions.h"

const float DELTAP = 0.000030518f;  // 2^(-16)

namespace delta {
const float window_length_sec = 0.025;
const float frame_length_sec = 0.010;
const int plp_order = 12;

PLP::PLP() {
  window_length_sec_ = window_length_sec;
  frame_length_sec_ = frame_length_sec;
  i_PlpOrd = plp_order;
  pf_spc = NULL;
  pf_wts = NULL;
  pf_lpc = NULL;
  pf_PLP = NULL;
  pclass_spc = NULL;
}

PLP::~PLP() {
  free(pf_spc);
  free(pf_wts);
  free(pf_lpc);
  free(pf_PLP);
  if (pclass_spc != NULL) {
    delete pclass_spc;
    pclass_spc = NULL;
  }
}

void PLP::set_window_length_sec(float window_length_sec) {
  window_length_sec_ = window_length_sec;
}

void PLP::set_frame_length_sec(float frame_length_sec) {
  frame_length_sec_ = frame_length_sec;
}

void PLP::set_plp_order(int plp_order) { i_PlpOrd = plp_order; }

int PLP::init_plp(int input_size, float sample_rate) {
  f_SamRat = sample_rate;
  i_WinLen = static_cast<int>(window_length_sec_ * f_SamRat);
  i_FrmLen = static_cast<int>(frame_length_sec_ * f_SamRat);
  i_NumFrm = (input_size - i_WinLen) / i_FrmLen + 1;
  i_FFTSiz = static_cast<int>(pow(2.0f, ceil(log2(i_WinLen))));
  i_NumFrq = i_FFTSiz / 2 + 1;
  i_NumCep = ceil(hz2bark(f_SamRat / 2)) + 1;
  f_LftExp = 0.6f;
  snprintf(s_FrqTyp, sizeof(s_FrqTyp), "bark");

  pf_spc = static_cast<float*>(malloc(sizeof(float) * i_NumFrq * i_NumFrm));
  pf_wts = static_cast<float*>(malloc(sizeof(float) * i_NumCep * i_NumFrq));
  pf_lpc =
      static_cast<float*>(malloc(sizeof(float) * (i_PlpOrd + 1) * i_NumFrm));
  pf_PLP =
      static_cast<float*>(malloc(sizeof(float) * (i_PlpOrd + 1) * i_NumFrm));

  pclass_spc = NULL;
  pclass_spc = new Spectrum();
  pclass_spc->init_spc(input_size, sample_rate);
  pclass_spc->set_is_fbank(true);
  pclass_spc->set_output_type(1);

  return 1;
}

int PLP::proc_plp(const float* mic_buf, int input_size) {
  int n, m, k;
  float tmp;
  float* ceps1 =
      static_cast<float*>(malloc(sizeof(float) * i_NumCep * i_NumFrm));
  float* ceps2 =
      static_cast<float*>(malloc(sizeof(float) * i_NumCep * i_NumFrm));
  float* ceps3 =
      static_cast<float*>(malloc(sizeof(float) * (i_PlpOrd + 1) * i_NumFrm));

  /* compute spectrum */
  pclass_spc->proc_spc(mic_buf, input_size);
  pclass_spc->get_spc(pf_spc);

  /* generate frequency scale warping */
  gen_warpweights(f_SamRat, i_NumCep, i_NumFrq, s_FrqTyp, i_FFTSiz, pf_wts);

  for (n = 0; n < i_NumFrm; n++) {
    for (k = 0; k < i_NumCep; k++) {
      tmp = 0.0;
      for (m = 0; m < i_NumFrq; m++) {
        tmp = tmp + pf_wts[k * i_NumFrq + m] * sqrt(pf_spc[n * i_NumFrq + m]);
      }
      tmp = pow(tmp, 2);
      ceps1[n * i_NumCep + k] = fmax(DELTAP, tmp);
    }
  }

  /* equalize loudness */
  do_EqlLoudness(f_SamRat, s_FrqTyp, i_NumCep, i_NumFrm, ceps1, ceps2);

  /* lpc analysis */
  compute_lpc(i_NumCep, i_NumFrm, i_PlpOrd, ceps2, pf_lpc);

  /* convert lpc to cepstra */
  do_lpc2cep(i_PlpOrd + 1, i_NumFrm, pf_lpc, ceps3);

  /* lift cepstrum */
  lift_cepstrum(f_LftExp, i_PlpOrd + 1, i_NumFrm, ceps3, pf_PLP);

  free(ceps1);
  free(ceps2);
  free(ceps3);
  return 1;
}

int PLP::get_plp(float* output) {
  int n, m;
  for (n = 0; n < i_NumFrm; n++) {
    for (m = 0; m < i_PlpOrd + 1; m++) {
      output[n * (i_PlpOrd + 1) + m] = pf_PLP[n * (i_PlpOrd + 1) + m];
    }
  }
  return 1;
}

int PLP::write_plp() {
  FILE* fp;
  fp = fopen("plp.txt", "w");
  int n, m;
  for (n = 0; n < i_NumFrm; n++) {
    for (m = 0; m < i_PlpOrd + 1; m++) {
      fprintf(fp, "%4.6f  ", pf_PLP[n * (i_PlpOrd + 1) + m]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
  return 1;
}
}  // namespace delta
