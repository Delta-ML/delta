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

#include "support_functions.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PI (3.141592653589793)

#define SWAP(a, b) \
  tempr = (a);     \
  (a) = (b);       \
  (b) = tempr

namespace delta {
/* compute mean */
float compute_mean(float* input, int i_size) {
  int n;
  float f_mean = 0.0;
  for (n = 0; n < i_size; n++) {
    f_mean = f_mean + input[n] / i_size;
  }
  return f_mean;
}

/* compute variance */
float compute_var(float* input, int i_size) {
  int n;
  float f_var = 0.0;
  float f_mean = compute_mean(input, i_size);

  for (n = 0; n < i_size; n++) {
    f_var = f_var + pow(input[n] - f_mean, 2.0) / i_size;
  }
  return f_var;
}

/* compute autocorrelation */
int compute_autoc(float* autoc, float* input, int i_size, int L) {
  if (i_size < L) {
    printf(
        "L should be less than i_size in autocorrelation computation, set "
        "L=i_size!\n");
    L = i_size;
  }
  int n, m;
  float f_mean = compute_mean(input, i_size);
  float f_var = compute_var(input, i_size);
  float tmp;

  for (m = 0; m <= L; m++) {
    autoc[m] = 0.0;
    for (n = 0; n < i_size - m; n++) {
      autoc[m] = autoc[m] + (input[n] - f_mean) * (input[n + m] - f_mean);
    }
    //  autoc[m] = (1.0/(i_size-m)) * autoc[m];
    //  autoc[m] = autoc[m] / f_var;
    if (m == 0) {
      tmp = autoc[0];
    }
    autoc[m] = autoc[m] / tmp;
  }
  return 0;
}

/* compute sign */
int compute_sign(float x) { return (x > 0) - (x < 0); }

/* generate window */
int gen_window(float* w, int L, char* typ) {
  float* pn = static_cast<float*>(malloc(sizeof(float) * L));
  int n;
  for (n = 0; n < L; n++) {
    pn[n] = 2 * PI * n / (L - 1);
  }

  if (strcmp(typ, "rect") == 0) {
    for (n = 0; n < L; n++) {
      w[n] = 1.0;
    }
  } else if (strcmp(typ, "tria") == 0) {
    for (n = 0; n < (L - 1) / 2; n++) {
      w[n] = n / ((L - 1) / 2);
    }
    for (n = (L - 1) / 2 + 1; n < L; n++) {
      w[n] = w[L - n - 1];
    }
  } else if (strcmp(typ, "hann") == 0) {
    for (n = 0; n < L; n++) {
      w[n] = 0.5 * (1 - cos(pn[n]));
    }
  } else if (strcmp(typ, "hamm") == 0) {
    for (n = 0; n < L; n++) {
      w[n] = 0.54 - 0.46 * cos(pn[n]);
    }
  } else if (strcmp(typ, "povey") == 0) {
    for (n = 0; n < L; n++){
      w[n] = pow(0.5 - 0.5 * cos(pn[n]), 0.85);
    }
  } else if (strcmp(typ, "blac") == 0) {
    for (n = 0; n < L; n++) {
      w[n] = 0.42 - 0.5 * cos(pn[n]) + 0.08 * cos(2 * pn[n]);
    }
  } else {
    printf("Window type not support!\n");
    return -1;
  }
  free(pn);
  return 0;
}

/* do pre-emphasis */
int do_preemphasis(float coef, float* buf, const float* input, int L) {
  /* a = filter([1 coef], 1, b) */
  if (coef == 0.0) {
    memcpy(buf, input, sizeof(float) * L);
    return 0;
  }

  int n;
  for (n = 0; n < L + 1; n++) {
    if (n == 0) {
      buf[n] = input[n] * 1.0;
    } else if (n == L) {
      buf[n] = input[n - 1] * (-1) * coef;
    } else {
      buf[n] = input[n] * 1.0 + input[n - 1] * (-1) * coef;
    }
  }

  return 0;
}

/* Hertz to Bark conversion */
float hz2bark(float f) {
  double z;
  z = 6.0 * asinh(static_cast<double>(f) / 600.0);
  return static_cast<float>(z);
}

/* Bark to Hertz conversion */
float bark2hz(float z) {
  double f;
  f = 600.0 * sinh(static_cast<double>(z) / 6.0);
  return static_cast<float>(f);
}

/* Hertz to Mel conversion */
float hz2mel(float f) {
  double z;
  z = 2595.0 * log10(1 + static_cast<double>(f) / 700.0);
  return static_cast<float>(z);
}

/* Mel to Hertz conversion */
float mel2hz(float z) {
  double f;
  f = 700.0 * (pow(10, static_cast<double>(z) / 2595.0) - 1);
  return static_cast<float>(f);
}

/* generate frequency scale warping */
int gen_warpweights(float fs, int ncep, int nfrq, char* ftyp, int nfft,
                    float* wts) {
  float bwidth = 1.0;
  float minfrq = 100.0;  /* lower bound in Hz */
  float maxfrq = fs / 2; /* upper bound in Hz */

  if (strcmp(ftyp, "bark") == 0) {
    float minbark = hz2bark(minfrq);
    float nyqbark = hz2bark(maxfrq) - minbark;
    float stpbark = nyqbark / (ncep - 1);
    int n, f;
    float f_bark_mid, lof, hif, tmp;
    float* binbark =
        static_cast<float*>(malloc(sizeof(float) * (nfft / 2 + 1)));

    for (n = 0; n < nfft / 2 + 1; n++) {
      binbark[n] = hz2bark(n * fs / nfft);
    }
    for (f = 0; f < ncep; f++) {
      f_bark_mid = minbark + f * stpbark;
      for (n = 0; n < nfft / 2 + 1; n++) {
        lof = (binbark[n] - f_bark_mid) / bwidth - 0.5;
        hif = (binbark[n] - f_bark_mid) / bwidth + 0.5;
        tmp = fmin(hif, -2.5 * lof);
        tmp = fmin(tmp, 0);
        wts[f * (nfft / 2 + 1) + n] = pow(10, tmp);
      }
    }

    free(binbark);
    return 0;
  } else if (strcmp(ftyp, "mel") == 0) {
    float minmel = hz2mel(minfrq);
    float maxmel = hz2mel(maxfrq);
    int n, f;
    float tmp, fss1, fss2, fss3, loslope, hislope;
    float* fftfrq = static_cast<float*>(malloc(sizeof(float) * (nfft / 2 + 1)));
    float* binfrq = static_cast<float*>(malloc(sizeof(float) * (ncep + 2)));

    for (n = 0; n < nfft / 2 + 1; n++) {
      fftfrq[n] = n * fs / nfft;
    }
    for (f = 0; f < ncep + 2; f++) {
      tmp = minmel + static_cast<float>(f) / (ncep + 1) * (maxmel - minmel);
      binfrq[f] = mel2hz(tmp);
    }

    for (f = 0; f < ncep; f++) {
      fss1 = binfrq[f];
      fss2 = binfrq[f + 1];
      fss3 = binfrq[f + 2];
      fss1 = fss2 + bwidth * (fss1 - fss2);
      fss3 = fss2 + bwidth * (fss3 - fss2);
      for (n = 0; n < nfft / 2 + 1; n++) {
        loslope = (fftfrq[n] - fss1) / (fss2 - fss1);
        hislope = (fss3 - fftfrq[n]) / (fss3 - fss2);
        tmp = fmin(loslope, hislope);
        wts[f * (nfft / 2 + 1) + n] = fmax(0.0f, tmp);
      }
    }

    free(fftfrq);
    free(binfrq);
    return 0;
  }
}

/* generate DCT matrix */
int gen_dctmatrix(int ncep, float* dctm) {
  int n, m;
  float tmp1, tmp2;
  tmp1 = static_cast<float>(PI / 2 / ncep);
  tmp2 = sqrt(2.0f / ncep);
  for (n = 0; n < ncep; n++) {
    for (m = 0; m < ncep; m++) {
      dctm[n * ncep + m] =
          cos(static_cast<float>(n) * (2 * m + 1) * tmp1) * tmp2;
    }
  }
  return 0;
}

/* lift cepstrum */
int lift_cepstrum(float lft_exp, int ncep, int nfrm, float* x, float* y) {
  int n, m;
  float liftwts;
  if (lft_exp <= 0) {
    memcpy(y, x, sizeof(float) * ncep * nfrm);
  } else {
    for (n = 0; n < ncep; n++) {
      if (n == 0) {
        liftwts = 1.0;
      } else {
        liftwts = pow(n, lft_exp);
      }
      for (m = 0; m < nfrm; m++) {
        y[m * ncep + n] = liftwts * x[m * ncep + n];
      }
    }
  }
  return 0;
}

/* cepstrum mean normalization */
int do_ceps_mean_norm(int ncep, int nfrm, float* x, float* y) {
  int n, m;
  float cmn;

  for (n = 0; n < ncep; n++) {
    if (n == 0) {
      for (m = 0; m < nfrm; m++) {
        y[m * ncep + n] = x[m * ncep + n];
      }
    } else {
      cmn = 0.0;
      for (m = 0; m < nfrm; m++) {
        cmn = cmn + x[m * ncep + n] / nfrm;
      }

      for (m = 0; m < nfrm; m++) {
        y[m * ncep + n] = x[m * ncep + n] - cmn;
      }
    }
  }

  return 0;
}

/* compute delta coefficients */
int compute_delta(int nw, int ncep, int nfrm, float* x, float* y) {
  if (nw % 2 == 0) /* make nw odd number */ {
    nw = nw + 1;
  }

  int hlen = (nw - 1) / 2;
  float* win = static_cast<float*>(malloc(sizeof(float) * nw));
  int n, m, k;
  float f_tmp, f_tmp2;
  int i_tmp, i_dist;

  for (n = 0; n < nw; n++) {
    win[n] = n - hlen; /* left-right flipped win */
  }

  for (n = 0; n < ncep; n++) {
    for (m = 0; m < nfrm; m++) {
      if (m == 0) {
        y[n * nfrm + m] = x[n * nfrm + m];
      } else if (m == nfrm - 1) {
        y[n * nfrm + m] = x[n * nfrm + m];
      } else {
        if (m < hlen) {
          i_dist = m;
        } else if (m > nfrm - 1 - hlen) {
          i_dist = nfrm - 1 - m;
        } else {
          i_dist = hlen;
        }
        f_tmp = 0.0;
        f_tmp2 = 0.0;
        for (k = 0; k <= 2 * i_dist; k++) {
          f_tmp = f_tmp + win[hlen - i_dist + k] * x[m - i_dist + k];
          f_tmp2 = f_tmp2 + (i_dist - k) * (i_dist - k);
        }
        y[n * nfrm + m] = f_tmp / f_tmp2;
      }
    }
  }

  free(win);
  return 0;
}

/* naive DFT */
int naive_dft(xcomplex* input, xcomplex* output, int inverse, int N) {
  float coef = (inverse == 1 ? 2 : -2) * PI;
  int n, m;
  float angle;
  for (int n = 0; n < N; n++) {
    output[n].r = 0.0;
    output[n].i = 0.0;
    for (m = 0; m < N; m++) {
      angle = coef * static_cast<float>(m * n % N) / N;
      output[n].r += input[m].r * cos(angle) - input[m].i * sin(angle);
      output[n].i += input[m].r * sin(angle) + input[m].i * cos(angle);
    }

    if (inverse == 1) {
      output[n] = complex_real_complex_mul(1.0f / N, output[n]);
    }
  }
  return 0;
}

/* do loudness equalization and cube root compression */
int do_EqlLoudness(float fs, char* ftyp, int ncep, int nfrm, float* x,
                   float* y) {
  int n, m;
  float tmp, tmp2, zmax;
  float* bandcfhz = static_cast<float*>(malloc(sizeof(float) * ncep));
  float* eql = static_cast<float*>(malloc(sizeof(float) * ncep));
  if (strcmp(ftyp, "bark") == 0) {
    zmax = hz2bark(fs);
    for (n = 0; n < ncep; n++) {
      tmp = static_cast<float>(n) * zmax / (ncep - 1);
      bandcfhz[n] = bark2hz(tmp);
    }
  } else if (strcmp(ftyp, "mel") == 0) {
    zmax = hz2mel(fs);
    for (n = 0; n < ncep; n++) {
      tmp = static_cast<float>(n) * zmax / (ncep - 1);
      bandcfhz[n] = mel2hz(tmp);
    }
  } else {
    printf("ftype not support in do_Eqlloudness\n");
    return -1;
  }

  /* Hynek's magic equal-loudness-curve formula */
  for (n = 0; n < ncep; n++) {
    tmp = pow(bandcfhz[n], 2);
    tmp2 = tmp + 1.6e5;
    eql[n] = pow(tmp / tmp2, 2) * (tmp + 1.44e6) / (tmp + 9.61e6);
  }

  /* weight the critical band */
  for (n = 0; n < nfrm; n++) {
    for (m = 0; m < ncep; m++) {
      tmp = eql[m] * x[n * ncep + m];
      y[n * ncep + m] = pow(tmp, 1.0f / 3.0f);
    }
  }

  /* replicate 1st and last band due to unreliability */
  for (n = 0; n < nfrm; n++) {
    y[n * ncep + 0] = y[n * ncep + 1];
    y[n * ncep + ncep - 1] = y[n * ncep + ncep - 2];
  }

  free(bandcfhz);
  free(eql);
  return 0;
}

/* convert LPC coeff into cepstrum */
int do_lpc2cep(int ncep, int nfrm, float* x, float* y) {
  int n, m, k;
  float tmp, sum;
  float* nx = static_cast<float*>(malloc(sizeof(float) * ncep * nfrm));

  /* 1st cep band is log(err) from Durbin */
  for (n = 0; n < nfrm; n++) {
    y[n * ncep + 0] = -log(x[n * ncep + 0]);
  }

  /* renormalize lpc coeff */
  for (n = 0; n < nfrm; n++) {
    nx[n * ncep + 0] = 1.0f;
    for (m = 1; m < ncep; m++) {
      nx[n * ncep + m] = x[n * ncep + m] / x[n * ncep + 0];
    }
  }

  for (n = 0; n < nfrm; n++) {
    for (m = 1; m < ncep; m++) {
      sum = 0.0f;
      for (k = 1; k <= m; k++) {
        sum = sum + (m - k) * nx[n * ncep + k] * y[n * ncep + m - k];
      }
      y[n * ncep + m] = -nx[n * ncep + m] - sum / m;
    }
  }

  free(nx);
  return 0;
}

/* Levinson durbin recursion */
int do_levinson(int pord, float* r, float* a) {
  float* as = static_cast<float*>(malloc(sizeof(float) * pord * pord));
  float* P = static_cast<float*>(malloc(sizeof(float) * pord));
  int n, m;
  float sum;

  /* for m = 1 */
  as[0] = -r[1] / r[0];
  P[0] = r[0] * (1 - pow(as[0], 2));
  /* for m = 2:pord */
  for (n = 1; n < pord; n++) {
    sum = 0.0f;
    for (m = 0; m < n; m++) {
      sum = sum + as[(n - 1) * pord + m] * r[n - m];
    }
    as[n * pord + n] = -(r[n + 1] + sum) / P[n - 1];
    for (m = 0; m < n; m++) {
      as[n * pord + m] = as[(n - 1) * pord + m] +
                         as[n * pord + n] * as[(n - 1) * pord + n - 1 - m];
    }
    P[n] = P[n - 1] * (1 - pow(as[n * pord + n], 2));
  }
  a[0] = 1.0f;
  for (n = 0; n < pord; n++) {
    a[n + 1] = as[(pord - 1) * pord + n];
  }

  free(as);
  free(P);
  return 0;
}

/* compute lpc coeff */
int compute_lpc(int ncep, int nfrm, int pord, float* x, float* y) {
  int n, m, nband = (ncep - 1) * 2;
  xcomplex* nx = static_cast<xcomplex*>(malloc(sizeof(xcomplex) * nband));
  xcomplex* fx = static_cast<xcomplex*>(malloc(sizeof(xcomplex) * nband));
  float* xr = static_cast<float*>(malloc(sizeof(float) * ncep));
  float* yr = static_cast<float*>(malloc(sizeof(float) * (pord + 1)));

  for (n = 0; n < nfrm; n++) {
    for (m = 0; m < ncep; m++) {
      nx[m].r = x[n * ncep + m];
      nx[m].i = 0.0f;
    }
    for (m = ncep; m < nband; m++) {
      nx[m].r = x[n * ncep + nband - m];
      nx[m].i = 0.0f;
    }

    naive_dft(nx, fx, 1, nband);

    for (m = 0; m < ncep; m++) {
      xr[m] = fx[m].r;
    }

    do_levinson(pord, xr, yr);

    for (m = 0; m < pord + 1; m++) {
      y[n * (pord + 1) + m] = yr[m];
    }
  }

  free(nx);
  free(fx);
  free(xr);
  free(yr);
  return 0;
}

/* Radix-2 DIT FFT */
/* isign=-1 ==> FFT, isign=1 ==> IFFT */
int dit_r2_fft(xcomplex* input, xcomplex* output, float* in_buf, int N, int isign) {
  float wtemp, wr, wpr, wpi, wi, theta;
  float tempr, tempi;
  int i = 0, j = 0, n = 0, k = 0, m = 0, istep, mmax;
  float* out_buf;
  float den;
  if (isign == -1)
    den = 1.0f;
  else
    den = static_cast<float>(N);

  for (i = 0; i < N; i++) {
    in_buf[i * 2] = input[i].r;
    in_buf[i * 2 + 1] = input[i].i;
  }

  out_buf = &in_buf[0] - 1;
  n = N * 2;
  j = 1;

  // do the bit-reversal
  for (i = 1; i < n; i += 2) {
    if (j > i) {
      SWAP(out_buf[j], out_buf[i]);
      SWAP(out_buf[j + 1], out_buf[i + 1]);
    }
    m = n >> 1;
    while (m >= 2 && j > m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }

  // calculate the FFT
  mmax = 2;
  while (n > mmax) {
    istep = mmax << 1;
    theta = isign * (2 * PI / mmax);
    wtemp = sin(0.5 * theta);
    wpr = -2.0 * wtemp * wtemp;
    wpi = sin(theta);
    wr = 1.0;
    wi = 0.0;
    for (m = 1; m < mmax; m += 2) {
      for (i = m; i <= n; i += istep) {
        j = i + mmax;
        tempr = wr * out_buf[j] - wi * out_buf[j + 1];
        tempi = wr * out_buf[j + 1] + wi * out_buf[j];
        out_buf[j] = out_buf[i] - tempr;
        out_buf[j + 1] = out_buf[i + 1] - tempi;
        out_buf[i] = out_buf[i] + tempr;
        out_buf[i + 1] = out_buf[i + 1] + tempi;
      }
      wtemp = wr;
      wr += wtemp * wpr - wi * wpi;
      wi += wtemp * wpi + wi * wpr;
    }
    mmax = istep;
  }

  for (i = 0; i < N; i++) {
    output[i].r = out_buf[i * 2 + 1] / den;
    output[i].i = out_buf[i * 2 + 2] / den;
  }
  return 0;
}

/* compute energy of frame */
float compute_energy(const float* input, int L){
    float energy = 0;
    for (int i = 0; i < L; i++){
        energy += input[i] * input[i];
    }
    return energy;
}

/* do pre_emphasis on frame */
int do_frame_preemphasis(float* input, float* output, int i_size, float coef){
    if (coef == 0.0){
        memcpy(output, input, sizeof(float) * i_size);
        return 0;
    }
    memcpy(output, input, sizeof(float) * i_size);
    for (int i = i_size - 1; i > 0; i--)
        output[i] -= coef * output[i-1];
    output[0] -= coef * output[0];
    return 0;
}

/* return subvector */
std::vector<BaseFloat> sub_vector(const std::vector<BaseFloat> &input, int begin, int length){
  std::vector<BaseFloat> res(length);
  for (int i = 0; i < length; i++){
    res[i] = input[i + begin];
  }
  return res;
}

std::vector<std::vector<BaseFloat>> sub_matrix(const std::vector<std::vector<BaseFloat>> &input,
                int row_begin, int n_rows, int col_begin, int n_cols){
    std::vector<std::vector<BaseFloat>> res;
    res.resize(n_rows);
    for (int i = 0; i < n_rows; i++){
      res[i].resize(n_cols);
      for (int j = 0; j < n_cols; j++)
        res[i][j] = input[row_begin + i][col_begin + j];
    }
    return res;
}

std::vector<BaseFloat> add_mat_vec(BaseFloat alpha, const std::vector<std::vector<BaseFloat>> &M,
                              const std::vector<BaseFloat> &v, BaseFloat beta){
    int row, col, length;
    row = M.size();
    col = M[0].size();
    length = v.size();
    std::vector<BaseFloat> res(row, 0);
    for (int i = 0; i < row; i++)
    {
      float tmp = 0.0;
      for (int j = 0; j < col; j++){
        tmp += M[i][j] * v[j];
      }
      res[i] = alpha * tmp + beta * res[i];
    }
    return res;
}

/*return gcd */
int  Gcd(int m, int n) {
  if (m == 0 || n == 0) {
    if (m == 0 && n == 0) {  // gcd not defined, as all integers are divisors.
      std::cerr << "Undefined GCD since m = 0, n = 0.";
    }
    return (m == 0 ? (n > 0 ? n : -n) : ( m > 0 ? m : -m));
    // return absolute value of whichever is nonzero
  }
  while (1) {
    m %= n;
    if (m == 0) return (n > 0 ? n : -n);
    n %= m;
    if (n == 0) return (m > 0 ? m : -m);
  }
}

int Lcm(int m, int n) {
  assert(m > 0 && n > 0);
  int gcd = Gcd(m, n);
  return gcd * (m/gcd) * (n/gcd);
}

BaseFloat VecVec(const vector<BaseFloat> &a, const vector<BaseFloat> &b){
  assert(a.size() == b.size());
  BaseFloat res = 0;
  for (int i = 0; i < a.size(); ++i)
    res += a[i] * b[i];
  return res;
}

/* do dither */
void do_dither(float* input, int i_size, float dither_value){
    if (dither_value == 0.0)
        return;
    RandomState rstate;
    for (int i = 0; i < i_size; i++)
        input[i] += RandGauss(&rstate) * dither_value;
}

int Rand(struct RandomState* state) {
#if !defined(_POSIX_THREAD_SAFE_FUNCTIONS)
  // On Windows and Cygwin, just call Rand()
  return rand();
#else
  if (state) {
    return rand_r(&(state->seed));
  } else {
    std::lock_guard<std::mutex> lock(_RandMutex);
    return rand();
  }
#endif
}

RandomState::RandomState() {
  seed = Rand() + 27437;
}

}  // namespace delta
