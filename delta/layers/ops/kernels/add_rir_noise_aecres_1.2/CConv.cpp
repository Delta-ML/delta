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

#include "CConv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

CConv::CConv() {
  buffer_len = 0;
  frm_len = 128;
  data_len = 1280;
  peakthld = 26000.0f;
  enableflag = 0x0a;
  apm_handle = NULL;
  inputdata = new short[frm_len * 2];
  bufferdata = new short[frm_len * 2];
  H = new double[RIR_LENGTH];

  normflag = 0;
}

CConv::CConv(int norm) {
  buffer_len = 0;
  frm_len = 128;
  data_len = 1280;
  peakthld = 26000.0f;
  enableflag = 0x0a;
  apm_handle = NULL;
  inputdata = new short[frm_len * 2];
  bufferdata = new short[frm_len * 2];
  H = new double[RIR_LENGTH];

  if (norm != 0) {
    normflag = 1;
  } else {
    normflag = 0;
  }
}

CConv::~CConv() {
  delete[] bufferdata;
  delete[] inputdata;
  delete[] H;
}

int CConv::SelectH(char *rir_list) {
  FILE *fplist = fopen(rir_list, "rt");
  if (NULL == fplist) {
    printf("Open rirlist file %s error \n", rir_list);
    return -1;
  }

  long int rir_num = 0;
  char rir_tmp_name[1024];
  while (fgets(rir_tmp_name, 1024, fplist)) {
    rir_num++;
  }
  fclose(fplist);
  if (NULL == fplist) {
    printf("Open rirlist file %s error AGAIN \n", rir_list);
    return -1;
  }

  int filter;
  srand((unsigned long)time(0));
  filter = abs(rand() * (rir_num - 1)) % rir_num;
  fplist = fopen(rir_list, "rt");
  if (fplist == NULL) {
    printf("Open rir list %s error \n", rir_list);
    return -2;
  }
  int ii = 0;
  while (fgets(rir_tmp_name, 1024, fplist)) {
    if (ii == filter) {
      break;
    }
    ii++;
  }
  rir_tmp_name[strlen(rir_tmp_name) - 1] = '\0';

  FILE *fprir = fopen(rir_tmp_name, "rb");
  if (fprir == NULL) {
    printf("Open rir file %s error \n", rir_tmp_name);
    return -3;
  }
  for (int kk = 0; kk < RIR_LENGTH; kk++) {
    double dtmp;
    double res = fread(&dtmp, sizeof(double), 1, fprir);
    H[kk] = dtmp;
  }

  return 0;
}

int CConv::ConvProcess(short *pOrigInputData, long lDataLength, double *ppRIR,
                       long lRIRLength, short *pOutputData) {
  if (pOrigInputData == NULL || ppRIR == NULL || pOutputData == NULL) {
    return -1;
  }
  if (lDataLength <= 0 || lRIRLength <= 0) {
    return -2;
  }

  float *pFloatData = new float[lDataLength];
  for (int ii = 0; ii < lDataLength; ii++) {
    pFloatData[ii] = 0.0;
  }

  int nFFTOrder = 15;
  int nFFTLength = 32768;

  nFFTLength = lRIRLength;
  if (lRIRLength & (lRIRLength - 1) == 0) {
    nFFTOrder = (int)(log(nFFTLength) / log(2)) + 1;
  } else {
    nFFTOrder = (int)(log(nFFTLength) / log(2)) + 2;
  }
  nFFTLength = (int)pow(2, nFFTOrder);

  int nBlockLength = nFFTLength / 2;
  COMPLEX *X = new COMPLEX[nFFTLength];
  COMPLEX *H = new COMPLEX[nFFTLength];

  for (int ii = 0; ii < nFFTLength; ii++) {
    X[ii].real = 0.0;
    X[ii].image = 0.0;
  }
  for (int ii = 0; ii < lRIRLength; ii++) {
    H[ii].real = ppRIR[ii];
    H[ii].image = 0.0;
  }
  for (int ii = lRIRLength; ii < nFFTLength; ii++) {
    H[ii].real = 0.0;
    H[ii].image = 0.0;
  }

  FFT(H, nFFTOrder);

  COMPLEX *XData = new COMPLEX[nFFTLength];
  long SegNum = (long)(lDataLength / nBlockLength);
  for (int ii = 0; ii < SegNum; ii++) {
    for (int jj = 0; jj < nBlockLength; jj++) {
      X[jj].real = X[jj + nBlockLength].real;
      X[jj].image = X[jj + nBlockLength].image;
    }

    for (int jj = 0; jj < nBlockLength; jj++) {
      X[jj + nBlockLength].real =
          (double)(pOrigInputData[ii * nBlockLength + jj]);
      X[jj + nBlockLength].image = 0.0;
    }

    for (int jj = 0; jj < nFFTLength; jj++) {
      XData[jj].real = X[jj].real;
      XData[jj].image = X[jj].image;
    }
    FFT(XData, nFFTOrder);
    for (int jj = 0; jj < nFFTLength; jj++) {
      double r, i;
      r = XData[jj].real * H[jj].real - XData[jj].image * H[jj].image;
      i = XData[jj].real * H[jj].image + XData[jj].image * H[jj].real;
      XData[jj].real = r;
      XData[jj].image = -i;
    }
    FFT(XData, nFFTOrder);
    for (int jj = 0; jj < nBlockLength; jj++) {
      pFloatData[ii * nBlockLength + jj] =
          XData[jj + nBlockLength].real / nFFTLength;
    }
  }
  if (SegNum * nBlockLength < lDataLength) {
    for (int jj = 0; jj < nBlockLength; jj++) {
      X[jj].real = X[jj + nBlockLength].real;
      X[jj].image = X[jj + nBlockLength].image;
    }

    for (int jj = 0; jj < lDataLength - SegNum * nBlockLength; jj++) {
      X[jj + nBlockLength].real =
          (double)(pOrigInputData[SegNum * nBlockLength + jj]);
      X[jj + nBlockLength].image = 0.0;
    }
    for (int jj = lDataLength - SegNum * nBlockLength; jj < nBlockLength;
         jj++) {
      X[jj + nBlockLength].real = 0.0;
      X[jj + nBlockLength].image = 0.0;
    }

    for (int jj = 0; jj < nFFTLength; jj++) {
      XData[jj].real = X[jj].real;
      XData[jj].image = X[jj].image;
    }
    FFT(XData, nFFTOrder);
    for (int jj = 0; jj < nFFTLength; jj++) {
      double r, i;
      r = XData[jj].real * H[jj].real - XData[jj].image * H[jj].image;
      i = XData[jj].real * H[jj].image + XData[jj].image * H[jj].real;
      XData[jj].real = r;
      XData[jj].image = -i;
    }
    FFT(XData, nFFTOrder);
    for (int jj = 0; jj < lDataLength - SegNum * nBlockLength; jj++) {
      pFloatData[SegNum * nBlockLength + jj] =
          XData[jj + nBlockLength].real / nFFTLength;
    }
  }

  double energy_in, energy_out;
  energy_in = 0.0;
  energy_out = 0.0;
  for (int ii = 0; ii < lDataLength; ii++) {
    energy_in += (double)pOrigInputData[ii] * (double)pOrigInputData[ii];
    energy_out += pFloatData[ii] * pFloatData[ii];
  }
  double beta = sqrt(energy_in / energy_out);
  if (normflag == 1) {
    for (int ii = 0; ii < lDataLength; ii++) {
      pFloatData[ii] *= (beta * 0.2);
    }
  } else {
    for (int ii = 0; ii < lDataLength; ii++) {
      pFloatData[ii] *= (beta * 1.0);
    }
  }

  float max_amplitude = 0.0;
  float alpha = 0.0;
  for (int ii = 0; ii < lDataLength; ii++) {
    if (fabs(pFloatData[ii]) > max_amplitude) {
      max_amplitude = fabs(pFloatData[ii]);
    }
  }
  if (max_amplitude > 32767) {
    alpha = 32767.0 / max_amplitude;
  } else {
    alpha = 1.0;
  }
  for (int ii = 0; ii < lDataLength; ii++) {
    pOutputData[ii] = (short)(pFloatData[ii] * alpha);
  }

  delete[] pFloatData;
  delete[] XData;
  delete[] X;
  delete[] H;

  return 0;
}
