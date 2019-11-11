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

#include "audio.h"

audio::audio(int nFs) { st = add_rir_noise_aecres_init(nFs); }

audio::~audio() { add_rir_noise_aecres_exit(st); }

int audio::audio_pre_proc(short* inputdata, int inputdata_length,
                          short* outputdata, int* outputdata_size,
                          bool if_add_rir, char* rir_filelist,
                          bool if_add_noise, char* noise_filelist,
                          float snr_min, float snr_max, bool if_add_aecres,
                          char* aecres_filelist) {
  int ret;
  ret = add_rir_noise_aecres_process(
      st, inputdata, inputdata_length, outputdata, outputdata_size, if_add_rir,
      rir_filelist, if_add_noise, noise_filelist, snr_min, snr_max,
      if_add_aecres, aecres_filelist);

  return ret;
}
