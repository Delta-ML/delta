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

#ifndef __CCONV_H_
#define __CCONV_H_
#include <math.h>
#include <stdio.h>
#include "BaseLib.h"
#include "typedefs_sh.h"
#define RIR_LENGTH 16000

class CConv {
 private:
 public:
  CConv(int normflag);
  CConv();
  ~CConv();

  void* apm_handle;
  short* inputdata;
  short* bufferdata;
  int buffer_len;
  int frm_len;
  int data_len;
  float peakthld;
  unsigned int enableflag;

  double* H;
  int ConvProcess(short* pOrigInputData, long lDataLength, double* ppRIR,
                  long lRIRLength, short* pOutputData);
  int SelectH(char* rir_list);

  int normflag;
};

#endif  //__CCONV_H_
