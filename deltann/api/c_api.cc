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

#include <string>
#include <vector>

#include "api/c_api.h"
#include "core/config.h"
#include "core/runtime.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

using delta::RuntimeConfig;
using delta::core::In;
using delta::core::Runtime;

ModelHandel DeltaLoadModel(const char* yaml_file) {
  std::string yaml = yaml_file;
  RuntimeConfig* rt_cfg = new RuntimeConfig(yaml_file);
  return static_cast<ModelHandel>(rt_cfg);
}

InferHandel DeltaCreate(ModelHandel model) {
  RuntimeConfig* rt_cfg = static_cast<RuntimeConfig*>(model);
  Runtime* rt = new Runtime(*rt_cfg);
  rt->warmup();
  return static_cast<InferHandel>(rt);
}

DeltaStatus DeltaSetInputs(InferHandel inf, Input* inputs, int num) {
  Runtime* rt = static_cast<Runtime*>(inf);
  std::vector<In> ins;
  for (int i = 0; i < num; ++i) {
    if (inputs[i].shape == NULL) {
      ins.push_back(In(inputs[i].graph_name, inputs[i].input_name,
                       inputs[i].ptr, inputs[i].size));
    } else {
      ins.push_back(In(inputs[i].graph_name, inputs[i].input_name,
                       inputs[i].shape, inputs[i].ndims, inputs[i].ptr,
                       inputs[i].size));
    }
  }
  rt->set_inputs(ins);
  return DeltaStatus::kDeltaOk;
}

DeltaStatus DeltaRun(InferHandel inf) {
  Runtime* rt = static_cast<Runtime*>(inf);
  rt->run();
  return DeltaStatus::kDeltaOk;
}

int DeltaGetOutputCount(InferHandel inf) {
  Runtime* rt = static_cast<Runtime*>(inf);
  return rt->get_output_num();
}

int DeltaGetOutputNumDims(InferHandel inf, int output_index) {
  Runtime* rt = static_cast<Runtime*>(inf);
  return rt->get_output_ndim(output_index);
}

int DeltaGetOutputDim(InferHandel inf, int output_index, int dim_index) {
  Runtime* rt = static_cast<Runtime*>(inf);
  return rt->get_output_dim(output_index, dim_index);
}

int DeltaGetOutputByteSize(InferHandel inf, int output_index) {
  Runtime* rt = static_cast<Runtime*>(inf);
  return rt->get_output_bytesize(output_index);
}

DeltaStatus DeltaCopyToBuffer(InferHandel inf, int output_index,
                              void* output_data, int output_data_byte_size) {
  Runtime* rt = static_cast<Runtime*>(inf);
  if (rt->copy_to_buffer(output_index, output_data, output_data_byte_size)) {
    return DeltaStatus::kDeltaError;
  }
  return DeltaStatus::kDeltaOk;
}

void DeltaDestroy(InferHandel inf) {
  Runtime* rt = static_cast<Runtime*>(inf);
  delete rt;
}

void DeltaUnLoadModel(ModelHandel model) {
  RuntimeConfig* rt_cfg = static_cast<RuntimeConfig*>(model);
  delete rt_cfg;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
