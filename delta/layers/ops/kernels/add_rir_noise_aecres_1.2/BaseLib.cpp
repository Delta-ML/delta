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

#include "BaseLib.h"
#include <math.h>
#include <stddef.h>
#include "typedefs_sh.h"

void FFT(COMPLEX *pFFTData, int nFFTOrder) {
  int n, i, nv2, j, k, le, l, le1, ip, nm1;
  COMPLEX t, u, w;

  n = 1;
  for (i = 0; i < (int)nFFTOrder; i++) {
    n = n * 2;
  }

  nv2 = n / 2;
  nm1 = n - 1;
  j = 1;

  for (i = 1; i <= nm1; i++) {
    if (i < j) {
      t.real = pFFTData[i - 1].real;
      t.image = pFFTData[i - 1].image;
      pFFTData[i - 1].real = pFFTData[j - 1].real;
      pFFTData[i - 1].image = pFFTData[j - 1].image;
      pFFTData[j - 1].real = t.real;
      pFFTData[j - 1].image = t.image;
    }

    k = nv2;

    while (k < j) {
      j -= k;
      k /= 2;
    }
    j += k;
  }

  le = 1;

  for (l = 1; l <= (int)nFFTOrder; l++) {
    le *= 2;
    le1 = le / 2;
    u.real = 1.0f;
    u.image = 0.0f;
    w.real = (float)cos(PI / le1);
    w.image = (float)-sin(PI / le1);

    for (j = 1; j <= le1; j++) {
      for (i = j; i <= n; i += le) {
        ip = i + le1;
        t.real =
            pFFTData[ip - 1].real * u.real - pFFTData[ip - 1].image * u.image;
        t.image =
            pFFTData[ip - 1].real * u.image + pFFTData[ip - 1].image * u.real;
        pFFTData[ip - 1].real = pFFTData[i - 1].real - t.real;
        pFFTData[ip - 1].image = pFFTData[i - 1].image - t.image;
        pFFTData[i - 1].real = t.real + pFFTData[i - 1].real;
        pFFTData[i - 1].image = t.image + pFFTData[i - 1].image;
      }

      t.real = u.real * w.real - u.image * w.image;
      t.image = u.image * w.real + u.real * w.image;
      u.real = t.real;
      u.image = t.image;
    }
  }
}
