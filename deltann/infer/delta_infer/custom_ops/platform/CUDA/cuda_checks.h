#ifndef _DELTA_INFER_CUDA_CHECK_H_
#define _DELTA_INFER_CUDA_CHECK_H_

#include <stdio.h>
#include <stdlib.h>

#ifdef WITH_CUDA
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cmath>
#include "cublas_v2.h"
#endif

namespace tensorflow {

#ifdef WITH_CUDA

#define ERROR_STATUS(Value) \
  case Value:               \
    return #Value

inline const char* CuBaseCublasGetErrorString(cublasStatus_t status) {
  switch (status) {
    ERROR_STATUS(CUBLAS_STATUS_SUCCESS);
    ERROR_STATUS(CUBLAS_STATUS_NOT_INITIALIZED);
    ERROR_STATUS(CUBLAS_STATUS_ALLOC_FAILED);
    ERROR_STATUS(CUBLAS_STATUS_INVALID_VALUE);
    ERROR_STATUS(CUBLAS_STATUS_ARCH_MISMATCH);
    ERROR_STATUS(CUBLAS_STATUS_MAPPING_ERROR);
    ERROR_STATUS(CUBLAS_STATUS_EXECUTION_FAILED);
    ERROR_STATUS(CUBLAS_STATUS_INTERNAL_ERROR);
    ERROR_STATUS(CUBLAS_STATUS_NOT_SUPPORTED);
    ERROR_STATUS(CUBLAS_STATUS_LICENSE_ERROR);
    default:
       break;
  }
  return "[ERROR] Cublas status value not known.";
}

inline const char* CuBaseCudnnGetErrorString(cudnnStatus_t status) {
  switch (status) {
    ERROR_STATUS(CUDNN_STATUS_SUCCESS);
    ERROR_STATUS(CUDNN_STATUS_NOT_INITIALIZED);
    ERROR_STATUS(CUDNN_STATUS_ALLOC_FAILED);
    ERROR_STATUS(CUDNN_STATUS_BAD_PARAM);
    ERROR_STATUS(CUDNN_STATUS_MAPPING_ERROR);
    ERROR_STATUS(CUDNN_STATUS_EXECUTION_FAILED);
    ERROR_STATUS(CUDNN_STATUS_INTERNAL_ERROR);
    ERROR_STATUS(CUDNN_STATUS_NOT_SUPPORTED);
    default:
       break;
  }
  return "[ERROR] Cudnn status value not known.";
}

template <typename ErrType>
inline const char* cubaseGetErrStr(ErrType error) {
  return "[ERROR] ErrType not known.";
}

template <>
inline const char* cubaseGetErrStr<cudaError_t>(cudaError_t status) {
  return cudaGetErrorString(status);
}

template <>
inline const char* cubaseGetErrStr<cublasStatus_t>(cublasStatus_t status) {
  return CuBaseCublasGetErrorString(status);
}

template <>
inline const char* cubaseGetErrStr<cudnnStatus_t>(cudnnStatus_t status) {
  return CuBaseCudnnGetErrorString(status);
}

#define cuda(function)                                                         \
  do {                                                                         \
    cudaError_t status = (cuda##function);                                     \
    if (cudaSuccess != (status)) {                                             \
      fprintf(stderr,                                                          \
              "[ERROR] CUDA Runtime Failure (line %d of file %s):\n\t"         \
              " %s returned 0x%x (%s)\n",                                      \
              __LINE__, __FILE__, #function, status, cubaseGetErrStr(status)); \
      exit(1);                                                                 \
    }                                                                          \
  } while (0);

#define cublas(function)                                                       \
  do {                                                                         \
    cublasStatus_t status = (cublas##function);                                \
    if (CUBLAS_STATUS_SUCCESS != (status)) {                                   \
      fprintf(stderr,                                                          \
              "[ERROR] Cublas Runtime Failure (line %d of file %s):\n\t"       \
              " %s returned 0x%x (%s)\n",                                      \
              __LINE__, __FILE__, #function, status, cubaseGetErrStr(status)); \
      exit(1);                                                                 \
    }                                                                          \
  } while (0);

#define cudnn(function)                                                        \
  do {                                                                         \
    cublasStatus_t status = (cudnn##function);                                 \
    if (CUDNN_STATUS_SUCCESS != (status)) {                                    \
      fprintf(stderr,                                                          \
              "[ERROR] Cudnn Runtime Failure (line %d of file %s):\n\t"        \
              " %s returned 0x%x (%s)\n",                                      \
              __LINE__, __FILE__, #function, status, cubaseGetErrStr(status)); \
      exit(1);                                                                 \
    }                                                                          \
  } while (0);

#endif
} /* namespace tensorflow */

#endif
