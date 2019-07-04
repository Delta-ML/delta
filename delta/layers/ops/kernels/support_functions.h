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

#ifndef DELTA_LAYERS_OPS_KERNELS_SUPPORT_FUNCTIONS_H_
#define DELTA_LAYERS_OPS_KERNELS_SUPPORT_FUNCTIONS_H_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "delta/layers/ops/kernels/complex_defines.h"

namespace delta {
/* compute mean */
float compute_mean(float* input, int i_size);

/* compute variance */
float compute_var(float* input, int i_size);

/* compute autocorrelation */
int compute_autoc(float* autoc, float* input, int i_size, int L);

/* compute sign */
int compute_sign(float x);

/* generate window */
int gen_window(float* w, int L, char* typ);

/* do pre-emphasis */
int do_preemphasis(float coef, float* buf, const float* input, int i_size);

/* Hertz to Bark conversion */
float hz2bark(float f);

/* Bark to Hertz conversion */
float bark2hz(float z);

/* Hertz to Mel conversion */
float hz2mel(float f);

/* Mel to Hertz conversion */
float mel2hz(float z);

/* generate frequency scale warping */
int gen_warpweights(float fs, int ncep, int nfrq, char* ftyp, int nfft,
                    float* wts);

/* generate DCT matrix */
int gen_dctmatrix(int ncep, float* dctm);

/* lift cepstrum */
int lift_cepstrum(float lft_exp, int ncep, int nfrm, float* x, float* y);

/* cepstrum mean normalization */
int do_ceps_mean_norm(int ncep, int nfrm, float* x, float* y);

/* compute delta coefficients */
int compute_delta(int nw, int ncep, int nfrm, float* x, float* y);

/* naive DFT */
int naive_dft(xcomplex* input, xcomplex* output, int inverse, int N);

/* do loudness equalization and cube root compression */
int do_EqlLoudness(float fs, char* ftyp, int ncep, int nfrm, float* x,
                   float* y);

/* convert lpc coeff into cepstrum */
int do_lpc2cep(int ncep, int nfrm, float* x, float* y);

/* Levinson durbin recursion */
int do_levinson(int pord, float* r, float* a);

/* compute lpc coeff */
int compute_lpc(int ncep, int nfrm, int pord, float* x, float* y);

/* radix-2 DIT FFT */
int dit_r2_fft(xcomplex* input, xcomplex* output, int N, int isign);

}  // namespace delta
#endif  // DELTA_LAYERS_OPS_KERNELS_SUPPORT_FUNCTIONS_H_
