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

#ifndef DELTA_LAYERS_OPS_KERNELS_COMPLEX_DEFINES_H_
#define DELTA_LAYERS_OPS_KERNELS_COMPLEX_DEFINES_H_

#include <math.h>

// define complex declare type
typedef struct FCOMPLEX {
  float r, i;
} fcomplex;
typedef struct ICOMPLEX {
  int r, i;
} icomplex;
typedef struct DCOMPLEX {
  double r, i;
} dcomplex;

// define fcomplex, float type
typedef fcomplex xcomplex;
typedef float xt;

// complex generation
inline xcomplex complex_gen(xt re, xt im) {
  xcomplex c;

  c.r = re;
  c.i = im;

  return c;
}

// complex conjugation
inline xcomplex complex_conjg(xcomplex z) {
  xcomplex c;

  c.r = z.r;
  c.i = -z.i;

  return c;
}

// complex abs
inline xt complex_abs(xcomplex z) {
  xt x, y, ans, temp;

  x = (xt)fabs(z.r);
  y = (xt)fabs(z.i);

  if (x == 0.0) {
    ans = y;
  } else if (y == 0.0) {
    ans = x;
  } else if (x > y) {
    temp = y / x;
    ans = x * (xt)sqrt(1.0 + temp * temp);
  } else {
    temp = x / y;
    ans = y * (xt)sqrt(1.0 + temp * temp);
  }

  return ans;
}

// complex number absolute value square
inline xt complex_abs2(xcomplex cp) {
  xt y;

  y = cp.r * cp.r + cp.i * cp.i;

  return (y);
}

// complex sqrt
inline xcomplex complex_sqrt(xcomplex z) {
  xcomplex c;
  xt x, y, w, r;

  if ((z.r == 0.0) && (z.i == 0.0)) {
    c.r = 0.0;
    c.i = 0.0;

    return c;
  } else {
    x = (xt)fabs(z.r);
    y = (xt)fabs(z.i);

    if (x >= y) {
      r = y / x;
      w = (xt)(sqrt(x) * sqrt(0.5 * (1.0 + sqrt(1.0 + r * r))));
    } else {
      r = x / y;
      w = (xt)(sqrt(y) * sqrt(0.5 * (r + sqrt(1.0 + r * r))));
    }

    if (z.r >= 0.0) {
      c.r = w;
      c.i = z.i / (2.0f * w);
    } else {
      c.i = (z.i >= 0) ? w : -w;
      c.r = z.i / (2.0f * c.i);
    }

    return c;
  }
}

// complex addition
inline xcomplex complex_add(xcomplex a, xcomplex b) {
  xcomplex c;

  c.r = a.r + b.r;
  c.i = a.i + b.i;

  return c;
}

// complex subtraction
inline xcomplex complex_sub(xcomplex a, xcomplex b) {
  xcomplex c;

  c.r = a.r - b.r;
  c.i = a.i - b.i;

  return c;
}

// complex multiplication
inline xcomplex complex_mul(xcomplex a, xcomplex b) {
  xcomplex c;

  c.r = a.r * b.r - a.i * b.i;
  c.i = a.i * b.r + a.r * b.i;

  return c;
}

// real and complex mutiplication
inline xcomplex complex_real_complex_mul(xt x, xcomplex a) {
  xcomplex c;

  c.r = x * a.r;
  c.i = x * a.i;

  return c;
}

// complex division
inline xcomplex complex_div(xcomplex a, xcomplex b) {
  xcomplex c;
  xt r, den;

  if (fabs(b.r) >= fabs(b.i)) {
    r = b.i / b.r;
    den = b.r + r * b.i;
    c.r = (a.r + r * a.i) / den;
    c.i = (a.i - r * a.r) / den;
  } else {
    r = b.r / b.i;
    den = b.i + r * b.r;
    c.r = (a.r * r + a.i) / den;
    c.i = (a.i * r - a.r) / den;
  }

  return c;
}

// complex division 2
inline xcomplex complex_div2(xcomplex a, xcomplex b) {
  xcomplex c;
  xt den;

  den = b.r * b.r + b.i * b.i;

  c.r = (a.r * b.r + a.i * b.i) / den;
  c.i = (a.i * b.r - a.r * b.i) / den;

  return c;
}

// complex number div real number
inline xcomplex complex_div_real(xcomplex cp, xt r) {
  xcomplex tmpcp;

  tmpcp.r = cp.r / r;
  tmpcp.i = cp.i / r;

  return (tmpcp);
}

// complex number averaging
inline xcomplex complex_avg_vec(xcomplex *cpVec, int cpVecLen) {
  int j;
  xcomplex tmpcp;

  tmpcp.r = 0.0;
  tmpcp.i = 0.0;

  for (j = 0; j < cpVecLen; j++) tmpcp = complex_add(tmpcp, cpVec[j]);

  tmpcp = complex_div_real(tmpcp, (xt)cpVecLen);

  return (tmpcp);
}

/* function implementations */
/*---------------------------------------------------
complex FIR filtering of 2nd dimension data
len    : Tap length        : in    : int
hat    : impulse response    : in    : xcomplex[][]
buf    : input buffer        : in    : xcomplex[][]
xout: output data        : out    : xcomplex[]
---------------------------------------------------*/
inline xcomplex complex_conv(int len, xcomplex *hat, xcomplex *buf) {
  int i;
  xcomplex z, xout;

  xout.r = xout.i = 0.0;
  for (i = 0; i < len; i++) {
    z = complex_mul(complex_conjg(hat[i]), buf[i]);
    xout = complex_add(xout, z);
  }
  return (xout);
}

/*---------------------------------------------------
CmplxDataPush
:  push complex data into 2nd dimension bufer
in    len    bufer length
xin    input data
renewal    buf    bufer
---------------------------------------------------*/
inline void complex_data_push(int len, xcomplex xin, xcomplex *buf) {
  int i;

  for (i = len - 1; i >= 1; i--) buf[i] = buf[i - 1];
  buf[0] = xin;
}

#endif  // DELTA_LAYERS_OPS_KERNELS_COMPLEX_DEFINES_H_
