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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include "api/c_api.h"

float time_run(InferHandel inf) {
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  DeltaRun(inf);
  clock_gettime(CLOCK_MONOTONIC, &end);
  return (end.tv_sec - start.tv_sec) +
         (end.tv_nsec - start.tv_nsec) / 1000000000.0;
}

typedef struct {
  void* ptr;
  int elems;
} Output;

struct DeltaModel {
  ModelHandel model_;
  InferHandel inf_;
  const char* model_yaml_;

  DeltaModel(const char* yaml) : model_yaml_(yaml) {
    model_ = DeltaLoadModel(model_yaml_);
    inf_ = DeltaCreate(model_);
  }

  ~DeltaModel() {
    DeltaDestroy(inf_);
    DeltaUnLoadModel(model_);
  }

  DeltaStatus SetInputs(float* buf, int bytes, const int* shape,
                        const int ndims) {
    Input ins[1];
    memset(ins, 0, sizeof(ins));

    ins[0].ptr = (void*)buf;
    ins[0].size = bytes;
    ins[0].graph_name = "spk-resnet";
    ins[0].input_name = "inputs:0";
    ins[0].shape = shape;
    ins[0].ndims = ndims;
    return DeltaSetInputs(inf_, ins, sizeof(ins) / sizeof(ins[0]));
  }

  DeltaStatus Run() { return DeltaRun(inf_); }

  int TimeRun() {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    Run();
    clock_gettime(CLOCK_MONOTONIC, &end);
    return (end.tv_sec - start.tv_sec) +
           (end.tv_nsec - start.tv_nsec) / 1000000000.0;
  }

  int GetOutputCnt() { return DeltaGetOutputCount(inf_); }

  DeltaStatus GetOutputs(float** buf, int* elems) {
    int out_num = GetOutputCnt();
    for (int i = 0; i < out_num; ++i) {
      int byte_size = DeltaGetOutputByteSize(inf_, i);
      *elems = byte_size / sizeof(float);
      float* tmp = (float*)malloc(byte_size);
      DeltaCopyToBuffer(inf_, i, (void*)(tmp), byte_size);
      buf[i] = tmp;
    }
    return kDeltaOk;
  }

  DeltaStatus GetOutputs(Output* outs) {
    int out_num = GetOutputCnt();
    for (int i = 0; i < out_num; ++i) {
      int byte_size = DeltaGetOutputByteSize(inf_, i);
      int elems = byte_size / sizeof(float);
      float* tmp = (float*)malloc(byte_size);
      DeltaCopyToBuffer(inf_, i, (void*)(tmp), byte_size);
      outs[i].ptr = tmp;
      outs[i].elems = elems;
    }
    return kDeltaOk;
  }
};

int main(int argc, char** argv) {
  const char* yaml_file = argv[1];

  DeltaModel m = DeltaModel(yaml_file);

  float avg = 0;
  int cnt = 1;
  for (int i = 0; i < cnt; i++) {
    int shape[] = {1, 260, 40, 1};
    shape[0] = i + 1;

    int bytes = sizeof(float);
    for (auto i : shape) bytes *= i;
    fprintf(stderr, "num elements %d\n", bytes);

    float* buf = (float*)malloc(bytes);
    memset(buf, 0, bytes);
    m.SetInputs(buf, bytes, shape, sizeof(shape) / sizeof(shape[0]));

    float dur = m.TimeRun();
    avg += dur;
    fprintf(stderr, "Duration %04f sec.\n", dur);
    free(buf);
    buf = nullptr;

    int out_num = m.GetOutputCnt();
    fprintf(stderr, "The output num is %d\n", out_num);

    // float** data = (float**)malloc(out_num * sizeof(nullptr));
    // memset(data, 0, out_num * sizeof(nullptr));

    // int elems = 0;
    // m.GetOutputs(data, &elems);
    // fprintf(stderr, "%d elements\n", elems);
    // fprintf(stderr, "%x ptr\n", data);

    // fprintf(stderr, "spk embeddings:\n");
    // for (int j = 0; j < elems; ++j) {
    //  fprintf(stderr, "%2.7f\t", data[0][j]);
    //  if ((j + 1) % 16 == 0) fprintf(stderr, "\n");
    //}
    // fprintf(stderr, "\n");
    // free(data[0]);
    // free(data);
    // data = nullptr;

    Output* outs = (Output*)malloc(out_num * sizeof(Output));
    m.GetOutputs(outs);
    for (int i = 0; i < out_num; i++) {
      fprintf(stderr, "Out Node %d\n", i);
      float* data = (float*)outs[i].ptr;
      for (int j = 0; j < outs[i].elems; j++) {
        fprintf(stderr, "%2.7f\t", data[j]);
        if ((j + 1) % 16 == 0) fprintf(stderr, "\n");
      }
      free(data);
    }
    free(outs);
  }
  fprintf(stderr, "Avg Duration %04f sec.\n", avg / cnt);
  return 0;
}
