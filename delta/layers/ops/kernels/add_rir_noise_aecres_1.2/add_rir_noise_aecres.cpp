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

#include "add_rir_noise_aecres.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "CAdd_All.h"

void* add_rir_noise_aecres_init(int nFs) {
  if (nFs != 16000) {
    printf("samplingrate error.\n");
    return NULL;
  }

  CAdd_All* MyAdd_All = new CAdd_All();

  return (void*)MyAdd_All;
}

int add_rir_noise_aecres_process(void* st, short* inputdata,
                                 int inputdata_length, short* outputdata,
                                 int* outputdata_size, bool if_add_rir,
                                 char* rir_filelist, bool if_add_noise,
                                 char* noise_filelist, float snr_min,
                                 float snr_max, bool if_add_aecres,
                                 char* aecres_filelist) {
  CAdd_All* MyAdd_All = (CAdd_All*)st;

  if (if_add_rir) {
    int ret;
    ret = MyAdd_All->add_rir(MyAdd_All->st_rir, inputdata, inputdata_length,
                             outputdata, outputdata_size, rir_filelist);
    if (ret < 0) {
      printf("add aecres error(%d).\n", ret);
      return ret;
    }
    memcpy(inputdata, outputdata, sizeof(short) * inputdata_length);
  }

  if (if_add_noise) {
    char filelist[1024];
    strcpy(filelist, noise_filelist);
    FILE* fplist = fopen(filelist, "rt");
    if (fplist == NULL) {
      printf("open noise filelist %s error \n", filelist);
      return -6;
    }
    long int file_num = 0;
    char file_tmp_name[1024];
    while (fgets(file_tmp_name, 1024, fplist)) {
      file_num++;
    }
    fclose(fplist);

    int file_idx;
    int loc_idx;
    file_idx = rand() % file_num;

    fplist = fopen(filelist, "rt");
    if (fplist == NULL) {
      printf("open noise filelist %s error AGAIN \n", filelist);
      return -7;
    }
    int kk = 0;
    while (fgets(file_tmp_name, 1024, fplist)) {
      if (kk == file_idx) {
        break;
      }
      kk++;
    }
    fclose(fplist);
    file_tmp_name[strlen(file_tmp_name) - 1] = '\0';

    FILE* fp = fopen(file_tmp_name, "rb");
    if (fp == NULL) {
      printf("Open %s Error.\n", file_tmp_name);
      return -4;
    }
    fseek(fp, 0, SEEK_END);
    long file_length = ftell(fp);
    file_length /= 2;
    rewind(fp);
    if (inputdata_length > file_length) {
      printf("input file too long.\n");
      memcpy(outputdata, inputdata, sizeof(short) * inputdata_length);
      outputdata_size[0] = inputdata_length;
    }
    long loc_max = file_length - inputdata_length - 2;
    loc_idx = rand() % loc_max;
    short* pnoise = new short[inputdata_length];
    fseek(fp, loc_idx * 2, SEEK_SET);
    short tmp = fread(pnoise, sizeof(short), inputdata_length, fp);
    fclose(fp);

    float SNR = snr_min;
    int r;
    r = rand() % ((int)snr_max - (int)snr_min + 1) + (int)snr_min;
    SNR = float(r);

    float signal_energy = 0.0;
    float noise_energy = 0.0;
    float beta, beta_tmp;
    for (int ii = 0; ii < inputdata_length; ii++) {
      signal_energy += (float)(inputdata[ii]) * (float)(inputdata[ii]);
      noise_energy += (float)(pnoise[ii]) * (float)(pnoise[ii]);
    }
    noise_energy *= 1.10;
    beta_tmp = signal_energy / noise_energy;
    beta = sqrt(beta_tmp / pow(10.0, SNR / 10));

    for (int ii = 0; ii < inputdata_length; ii++) {
      float tmp = (float)(inputdata[ii]) + (float)(pnoise[ii]) * beta;
      if (tmp > 32767.0) {
        outputdata[ii] = 32767;
      } else if (tmp < -32768.0) {
        outputdata[ii] = -32768;
      } else {
        outputdata[ii] = (short)tmp;
      }
    }
    outputdata_size[0] = inputdata_length;

    memcpy(inputdata, outputdata, sizeof(short) * inputdata_length);
    delete[] pnoise;
  }

  if (if_add_rir == false && if_add_noise == false) {
    memcpy(outputdata, inputdata, sizeof(short) * inputdata_length);
    outputdata_size[0] = inputdata_length;
  }

  return 0;
}

void add_rir_noise_aecres_exit(void* st) {
  CAdd_All* MyAdd_All = (CAdd_All*)st;
  delete MyAdd_All;
}
