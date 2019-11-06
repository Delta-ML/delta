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

#ifndef DELTANN_API_C_API_H_
#define DELTANN_API_C_API_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define DELTA_CAPI_EXPORT __attribute__((visibility("default")))

#define MAX_NAME_SIZE 30

typedef enum { kDeltaOk = 0, kDeltaError = 1 } DeltaStatus;

typedef struct {
  const void* ptr;
  int size;  // byte_size;
  const char* input_name;
  const char* graph_name;
  const int* shape;
  int ndims;
} Input;

typedef void* ModelHandel;
typedef void* InferHandel;

// load model
DELTA_CAPI_EXPORT extern ModelHandel DeltaLoadModel(const char* yaml_file);

// creater Interpreter
DELTA_CAPI_EXPORT extern InferHandel DeltaCreate(ModelHandel model);

// set input
DELTA_CAPI_EXPORT extern DeltaStatus DeltaSetInputs(InferHandel inf, Input* in,
                                                    int elem_num);

// run
DELTA_CAPI_EXPORT extern DeltaStatus DeltaRun(InferHandel inf);

// Returns the number of output.
DELTA_CAPI_EXPORT extern int DeltaGetOutputCount(InferHandel inf);

// Returns the number of dimensions that the output has.
DELTA_CAPI_EXPORT extern int DeltaGetOutputNumDims(InferHandel inf,
                                                   int output_index);

// Returns the length of the output in the "dim_index" dimension.
DELTA_CAPI_EXPORT extern int DeltaGetOutputDim(InferHandel inf,
                                               int output_index, int dim_index);

DELTA_CAPI_EXPORT extern int DeltaGetOutputByteSize(InferHandel inf,
                                                    int output_index);

// Copies to the provided output buffer from the tensor's buffer.
// REQUIRES: output_data_size == DeltaGetOutputByteSize()
DELTA_CAPI_EXPORT extern DeltaStatus DeltaCopyToBuffer(
    InferHandel inf, int output_index, void* output_data,
    int output_data_byte_size);

// delete inference
DELTA_CAPI_EXPORT extern void DeltaDestroy(InferHandel inf);

// unloade model
DELTA_CAPI_EXPORT extern void DeltaUnLoadModel(ModelHandel model);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // DELTANN_API_C_API_H_
