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

#include "api/c_api.h"

int main(int argc, char** argv) {
  // const char* yaml_file = "model.yaml";
  const char* yaml_file = argv[1];

  ModelHandel model = DeltaLoadModel(yaml_file);
  InferHandel inf = DeltaCreate(model);

  int in_num = 1;
  Input ins[1];
  const char* text = "懂了客服";
  ins[0].ptr = (void*)(&text);
  ins[0].size = sizeof(text);
  ins[0].input_name = "input_sentence";
  ins[0].graph_name = "default";

  DeltaSetInputs(inf, ins, in_num);

  DeltaRun(inf);

  int out_num = DeltaGetOutputCount(inf);
  fprintf(stderr, "The output num is %d\n", out_num);
  for (int i = 0; i < out_num; ++i) {
    int byte_size = DeltaGetOutputByteSize(inf, i);
    fprintf(stderr, "The %d output byte size is %d\n", i, byte_size);

    float* data = (float*)malloc(byte_size);
    DeltaCopyToBuffer(inf, i, (void*)data, byte_size);

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
