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

#ifndef __CECHO_H_
#define __CECHO_H_
#include <math.h>
#include <stdio.h>

class CEcho {
 private:
 public:
  CEcho(int f, float echo_snr_min, float echo_snr_max, float echo_ratio);
  ~CEcho();

  int process(short* inputdata, int inputdata_length, short* outputdata,
              int* outputdata_size, char* filelist);

 public:
  int nFs;
  int ahead;
  int tail;
  float snr_min;
  float snr_max;
  float snr_ratio;
};

#endif  //__CECHO_H_
