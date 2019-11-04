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
  const char* yaml_file = "../dpl/model/model.yaml";

  ModelHandel model = DeltaLoadModel(yaml_file);
  InferHandel inf = DeltaCreate(model);

  int in_num = 2;
  Input ins[2] = {0};
  int text = 1;
  ins[0].ptr = reinterpret_cast<void*>(&text);
  ins[0].size = 1;
  ins[0].input_name = "texts";
  ins[0].graph_name = "emotion";

  int size = 3000 * 40 * 3;
  float* buffer = reinterpret_cast<float*>(malloc(size * sizeof(float)));
  ins[1].ptr = reinterpret_cast<void*>(buffer);
  ins[1].size = size;

  ins[1].input_name = "inputs";
  ins[1].graph_name = "emotion";

  DeltaSetInputs(inf, ins, in_num);

  int out_num = DeltaGetOutputCount(inf);

  int err = DeltaRun(inf);
  if (err != 0) {
    fprintf(stderr, "Infer error\n");
    goto end;
  }

  fprintf(stderr, "The output num is %d\n", out_num);
  for (int i = 0; i < out_num; ++i) {
    int byte_size = DeltaGetOutputByteSize(inf, i);
    fprintf(stderr, "The %d output byte size is %d\n", i, byte_size);

    float* data = reinterpret_cast<float*>(malloc(byte_size));
    DeltaCopyToBuffer(inf, i, reinterpret_cast<void*>(data), byte_size);

    int num = byte_size / sizeof(float);
    for (int j = 0; j < num; ++j) {
      fprintf(stderr, "score is %f\n", data[j]);
    }
    free(data);
  }

end:
  DeltaDestroy(inf);
  DeltaUnLoadModel(model);

  free(buffer);

  return 0;
}
