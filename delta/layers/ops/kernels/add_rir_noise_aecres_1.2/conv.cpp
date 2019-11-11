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

#include "conv.h"
#include <stdio.h>
#include <string.h>
#include "CConv.h"

void* conv_init(int nFs, int normflag) {
  if (nFs != 16000 && nFs != 8000) {
    printf("SamplingRate Error!\n");
    return NULL;
  }

  CConv* MyConv = new CConv(normflag);

  return (void*)MyConv;
}

int conv_process(void* st, short* inputdata, int inputdata_length,
                 short* outputdata, int* outputdata_size, char* rir_list) {
  CConv* MyConv = (CConv*)st;

  int ret;
  ret = MyConv->SelectH(rir_list);
  if (ret < 0) {
    return ret;
  }
  MyConv->ConvProcess(inputdata, (long)inputdata_length, MyConv->H, RIR_LENGTH,
                      outputdata);
  outputdata_size[0] = inputdata_length;

  return 0;
}

void conv_exit(void* st) {
  CConv* MyConv = (CConv*)st;
  delete MyConv;
}
