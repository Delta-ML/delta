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
#include <vector>

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

template <class T>
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

  std::size_t NumElems(const std::vector<int> shape) {
    std::size_t size = 1;
    for (auto i : shape) size *= i;
    return size;
  }

  std::size_t Bytes(const std::vector<int> shape) {
    std::size_t bytes = sizeof(T);
    return bytes * this->NumElems(shape);
  }

  T* AllocInputs(const std::vector<int> shape) {
    std::size_t bytes = this->Bytes(shape);
    T* buf = (T*)malloc(bytes);
    memset(buf, 0, bytes);
    return buf;
  }

  DeltaStatus SetInputs(T* buf, const std::vector<int> shape) {
    return this->SetInputs(buf, this->Bytes(shape), shape.data(), shape.size());
  }

  DeltaStatus SetInputs(T* buf, int bytes, const int* shape, const int ndims) {
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

  DeltaStatus GetOutputs(T** buf, int* elems) {
    int out_num = GetOutputCnt();
    for (int i = 0; i < out_num; ++i) {
      int byte_size = DeltaGetOutputByteSize(inf_, i);
      *elems = byte_size / sizeof(T);
      T* tmp = (T*)malloc(byte_size);
      DeltaCopyToBuffer(inf_, i, (void*)(tmp), byte_size);
      buf[i] = tmp;
    }
    return kDeltaOk;
  }

  DeltaStatus GetOutputs(Output* outs) {
    int out_num = GetOutputCnt();
    for (int i = 0; i < out_num; ++i) {
      int byte_size = DeltaGetOutputByteSize(inf_, i);
      int elems = byte_size / sizeof(T);
      T* tmp = (T*)malloc(byte_size);
      DeltaCopyToBuffer(inf_, i, (void*)(tmp), byte_size);
      outs[i].ptr = tmp;
      outs[i].elems = elems;
    }
    return kDeltaOk;
  }

  Output* AllocOutputs(int cnt) {
    int bytes = cnt * sizeof(Output);
    Output* outs = (Output*)malloc(bytes);
    memset(outs, 0, bytes);
    return outs;
  }

  Output* AllocOutputs() {
    int out_num = GetOutputCnt();
    return AllocOutputs(out_num);
  }

  DeltaStatus FreeOutputs(Output* outs, int cnt) {
    for (int i = 0; i < cnt; i++) {
      T* data = (T*)outs[i].ptr;
      free(data);
    }
    free(outs);
    return kDeltaOk;
  }

  DeltaStatus FreeOutputs(Output* outs) {
    int out_num = GetOutputCnt();
    return FreeOutputs(outs, out_num);
  }
};

template struct DeltaModel<float>;

int main(int argc, char** argv) {
  const char* yaml_file = argv[1];

  DeltaModel<float> m(yaml_file);

  float avg = 0;
  int cnt = 1000;
  for (int i = 0; i < cnt; i++) {
    std::vector<int> shape = {1, 260, 40, 1};
    shape[0] = i + 1;

    float* buf = m.AllocInputs(shape);
    auto nelems = m.NumElems(shape);
    for (auto i = 0; i < nelems; i++) {
      buf[i] = 0;
    }
    m.SetInputs(buf, shape);

    float dur = m.TimeRun();
    fprintf(stderr, "Duration %04f sec.\n", dur);
    avg += dur;

    free(buf);
    buf = nullptr;

    Output* outs = m.AllocOutputs();
    m.GetOutputs(outs);

    int out_num = m.GetOutputCnt();
    fprintf(stderr, "The output num is %d\n", out_num);
    for (int i = 0; i < out_num; i++) {
      fprintf(stderr, "Out Node %d\n", i);
      float* data = (float*)outs[i].ptr;
      for (int j = 0; j < outs[i].elems; j++) {
        fprintf(stderr, "%2.7f\t", data[j]);
        if ((j + 1) % 16 == 0) fprintf(stderr, "\n");
      }
    }

    m.FreeOutputs(outs);
  }
  fprintf(stderr, "Avg Duration %04f sec.\n", avg / cnt);
  return 0;
}
