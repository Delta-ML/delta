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

#ifndef __ADD_ECHO_H_
#define __ADD_ECHO_H_

#ifdef __cplusplus
extern "C" {
#endif

void* add_echo_init(int nFs, float echo_snr_min, float echo_snr_max,
                    float echo_ratio);

int add_echo_process(void* st, short* inputdata, int inputdata_length,
                     short* outputdata, int* outputdata_size, char* filelist);

void add_echo_exit(void* st);

#ifdef __cplusplus
}
#endif
#endif  //__ADD_ECHO_H_
