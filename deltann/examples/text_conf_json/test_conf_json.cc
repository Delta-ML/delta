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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <vector>
#include "json/json.h"

using namespace std;

#include "api/c_api.h"
#include <iostream>

float time_run(InferHandel inf) {
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  DeltaRun(inf);
  clock_gettime(CLOCK_MONOTONIC, &end);
  return (end.tv_sec - start.tv_sec) +
         (end.tv_nsec - start.tv_nsec) / 1000000000.0;
}

string write_Json(){
  Json::Value root;
  Json::Value input_idx;
  Json::Value input_pad;
  Json::Value input_seg;

  for(int i=0; i<512; i++){
    input_idx[i] = 0;
  }
  input_idx[0] = 102;
  input_idx[1] = 873;
  input_idx[2] = 1358;
  input_idx[3] = 3868;
  input_idx[4] = 1417;
  input_idx[5] = 1963;
  input_idx[6] = 4639;

  for(int i=0; i<512; i++){
    input_pad[i] = 0;
  }
  input_pad[0] = 1;
  input_pad[1] = 1;
  input_pad[2] = 1;
  input_pad[3] = 1;
  input_pad[4] = 1;
  input_pad[5] = 1;
  input_pad[6] = 1;

  for(int i=0; i<512; i++){
    input_seg[i] = 0;
  }
  input_seg[0] = 0;
  input_seg[1] = 0;
  input_seg[2] = 0;
  input_seg[3] = 0;
  input_seg[4] = 0;
  input_seg[5] = 1;
  input_seg[6] = 1; 

  root["input_ids"] = input_idx;
  root["input_mask"] = input_pad;
  root["input_seg"] = input_seg;

  root.toStyledString();
  string out = root.toStyledString();
  return out;
}


int main(int argc, char** argv) {
  // const char* yaml_file = "model.yaml";
  const char* yaml_file = argv[1];
  string input_json = write_Json();

  ModelHandel model = DeltaLoadModel(yaml_file);
  InferHandel inf = DeltaCreate(model);
  DeltaSetJsonInputs(inf, input_json.c_str());
  cout<<"finish parse"<<endl;

  // DeltaRun(inf);
  float dur = time_run(inf);
  fprintf(stderr, "Duration %04f sec.\n", dur);

  int out_num = DeltaGetOutputCount(inf);
  fprintf(stderr, "The output num is %d\n", out_num);
  for (int i = 0; i < out_num; ++i) {
    int byte_size = DeltaGetOutputByteSize(inf, i);
    fprintf(stderr, "The %d output size is %d (bytes).\n", i, byte_size);

    float* data = reinterpret_cast<float*>(malloc(byte_size));
    DeltaCopyToBuffer(inf, i, reinterpret_cast<void*>(data), byte_size);

    int num = byte_size / sizeof(float);
    for (int j = 0; j < num; ++j) {
      fprintf(stderr, "score is %f\n", data[j]);
    }
    free(data);
  }
  DeltaDestroy(inf);
  DeltaUnLoadModel(model);

  return 0;
}
