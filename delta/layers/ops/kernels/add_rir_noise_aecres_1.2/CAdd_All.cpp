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

#include "CAdd_All.h"
#include "conv.h"

CAdd_All::CAdd_All() { st_rir = conv_init(16000, 0); }

CAdd_All::~CAdd_All() { conv_exit(st_rir); }

int CAdd_All::add_rir(void* st, short* inputdata, int inputdata_length,
                      short* outputdata, int* outputdata_size, char* filelist) {
  int ret;
  ret = conv_process(st, inputdata, inputdata_length, outputdata,
                     outputdata_size, filelist);
  return ret;
}
