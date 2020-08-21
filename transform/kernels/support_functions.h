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

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <limits>
#include <iostream>
#include <cmath>
#include <vector>
#include <memory>
#include <chrono>

#include "complex_defines.h"

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

#ifndef M_2PI
#define M_2PI 6.283185307179586476925286766559005
#endif

using namespace std;

namespace delta {
typedef float  BaseFloat;
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
int dit_r2_fft(xcomplex* input, xcomplex* output, float* in_buf, int N, int isign);

/* compute energy of frame */
float compute_energy(const float* input, int L);

/* do frame_pre_emphasis */
int do_frame_preemphasis(float* input, float* output, int i_size, float coef);

/* add dither */
void do_dither(float* input, int i_size, float dither_value);

/* return subvector */
std::vector<BaseFloat> sub_vector(const std::vector<BaseFloat> &input, int begin, int length);

std::vector<std::vector<BaseFloat>> sub_matrix(const std::vector<std::vector<BaseFloat>> &input,
                int row_begin, int n_rows, int col_begin, int n_cols);

std::vector<BaseFloat> add_mat_vec(BaseFloat alpha, const std::vector<std::vector<BaseFloat>> &M,
                              const std::vector<BaseFloat> &v, BaseFloat beta);
/* return gcd */
int Gcd(int m, int n);

/*return lcm */
int Lcm(int m, int n);

BaseFloat VecVec(const vector<BaseFloat> &a, const vector<BaseFloat> &b);

static inline bool ApproxEqual(BaseFloat a, BaseFloat b,
                               BaseFloat relative_tolerance = 0.001) {
  // a==b handles infinities.
  if (a == b) return true;
  BaseFloat diff = std::fabs(a-b);
  if (diff == std::numeric_limits<BaseFloat>::infinity()
      || diff != diff) return false;  // diff is +inf or nan.
  return (diff <= (relative_tolerance * (std::fabs(a)+std::fabs(b))));
}

// Returns a random integer between 0 and RAND_MAX, inclusive
int Rand(struct RandomState* state = NULL);

// State for thread-safe random number generator
struct RandomState {
  RandomState();
  unsigned seed;
};

inline double Log(double x) { return log(x); }

/// Returns a random number strictly between 0 and 1.
inline float RandUniform(struct RandomState* state = NULL) {
  return static_cast<float>((Rand(state) + 1.0) / (RAND_MAX+2.0));
}

inline float RandGauss(struct RandomState* state = NULL) {
  return static_cast<float>(sqrtf (-2 * Log(RandUniform(state)))
                            * cosf(2*M_PI*RandUniform(state)));
}

}  // namespace delta
#endif  // DELTA_LAYERS_OPS_KERNELS_SUPPORT_FUNCTIONS_H_
