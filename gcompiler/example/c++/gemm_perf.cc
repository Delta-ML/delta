#include "absl/strings/match.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/protobuf.h"

#include <sys/time.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#ifdef WITH_CUDA
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include "delta_infer/custom_ops/platform/CUDA/cuda_checks.h"
#endif

using namespace tensorflow;

double diffTime(timeval start, timeval end) {
  return (end.tv_sec - start.tv_sec) * 1000 +
         (end.tv_usec - start.tv_usec) * 0.001;
}

int main(int argc, const char** argv) {
  if (argc < 5) {
    fprintf(
        stderr,
        "ERROR: need paramter batch_size, seq_len, head_num, size_per_head\n");
    exit(1);
  }

  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Detect Device %s\n", prop.name);
  FILE* fd = fopen("./gemm_config.in", "wb");
  if (fd == NULL) {
    printf("Cannot write to file gemm_config.in\n");
    return 0;
  }

  const int batch_size = atoi(argv[1]);
  const int seq_len = atoi(argv[2]);
  const int head_num = atoi(argv[3]);
  const int size_per_head = atoi(argv[4]);

#if (CUDART_VERSION >= 10000)  /// cuda > 10.0
  const int gemm_num = 5;

#else
  const int gemm_num = 3;
#endif
  int M[gemm_num];
  int N[gemm_num];
  int K[gemm_num];
  int batchCount[gemm_num] = {0};
  for (int i = 0; i < gemm_num; i++) {
    batchCount[i] = 1;
  }
  char mess[gemm_num][256];

  // gemm1
  M[0] = batch_size * seq_len;
  K[0] = head_num * size_per_head;
  N[0] = K[0];
  strcpy(mess[0], "from_tensor * weightQ/K/V, attr * output_kernel");

  // gemm2
  M[1] = M[0];
  K[1] = K[0];
  N[1] = 4 * N[0];
  strcpy(mess[1], "attr_output * inter_kernel");

  // gemm3
  M[2] = M[0];
  K[2] = 4 * K[0];
  N[2] = N[0];
  strcpy(mess[2], "inter_matmul * output_kernel");

  M[3] = seq_len;
  N[3] = seq_len;
  K[3] = size_per_head;
  batchCount[3] = batch_size * head_num;
  strcpy(mess[3], "attention batched Gemm1");

  M[4] = seq_len;
  N[4] = size_per_head;
  K[4] = seq_len;
  batchCount[4] = batch_size * head_num;
  strcpy(mess[4], "attention batched Gemm2");

  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);

  typedef float T;
  cudaDataType_t AType = CUDA_R_32F;
  cudaDataType_t BType = CUDA_R_32F;
  cudaDataType_t CType = CUDA_R_32F;
  cudaDataType_t computeType = CUDA_R_32F;
  const int ites = 1000;
  struct timeval start, end;
  int startAlgo = (int)CUBLAS_GEMM_DEFAULT;
#if (CUDART_VERSION >= 10000)  /// cuda > 10.0
  int endAlgo = (int)CUBLAS_GEMM_ALGO23;
#else
  int endAlgo = (int)CUBLAS_GEMM_ALGO17;
#endif
  T alpha = (T)1.0f;
  T beta = (T)0.0f;

  printf("***FP32 Gemm Testing***\n");
  for (int i = 0; i < gemm_num; ++i) {
    int m = M[i], n = N[i], k = K[i];
    printf("\n-----------------------------\n");
    printf("GEMM test %d: [M: %d, K: %d, N: %d] %s\n", i, m, k, n, mess[i]);
    T* d_A;
    T* d_B;
    T* d_C;
    cuda(Malloc((void**)&d_A, sizeof(T) * m * k * batchCount[i]));
    cuda(Malloc((void**)&d_B, sizeof(T) * k * n * batchCount[i]));
    cuda(Malloc((void**)&d_C, sizeof(T) * m * n * batchCount[i]));

    float exec_time = 99999.0f;
    int fast_algo = 0;
    for (int algo = startAlgo; algo <= endAlgo; algo++) {
      cublasStatus_t status;
      cudaDeviceSynchronize();
      gettimeofday(&start, NULL);
      for (int ite = 0; ite < ites; ++ite) {
        if (i < 3) {
          status = cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m,
                                k, &alpha, d_B, BType, n, d_A, AType, k, &beta,
                                d_C, CType, n, computeType,
                                static_cast<cublasGemmAlgo_t>(algo));
        } else if (i == 3) {
#if (CUDART_VERSION >= 10000)  /// cuda > 10.0
          status = cublasGemmStridedBatchedEx(
              cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, seq_len, seq_len,
              size_per_head, &alpha, d_B, BType, size_per_head,
              seq_len * size_per_head, d_A, AType, size_per_head,
              seq_len * size_per_head, &beta, d_C, CType, seq_len,
              seq_len * seq_len, batch_size * head_num, computeType,
              static_cast<cublasGemmAlgo_t>(algo));
#endif
        } else {
#if (CUDART_VERSION >= 10000)  /// cuda > 10.0
          status = cublasGemmStridedBatchedEx(
              cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, size_per_head, seq_len,
              seq_len, &alpha, d_B, BType, size_per_head,
              seq_len * size_per_head, d_A, AType, seq_len, seq_len * seq_len,
              &beta, d_C, CType, size_per_head, seq_len * size_per_head,
              batch_size * head_num, computeType,
              static_cast<cublasGemmAlgo_t>(algo));
#endif
        }
      }
      cudaDeviceSynchronize();
      gettimeofday(&end, NULL);
      if (status == CUBLAS_STATUS_SUCCESS) {
        printf("\\__algo_%d costs %.3fms \n", algo,
               diffTime(start, end) / ites);
        if (diffTime(start, end) / ites < exec_time) {
          exec_time = diffTime(start, end) / ites;
          fast_algo = algo;
        }
      }
    }
    printf("   @==> fast_algo %d costs %.3f ms\n", fast_algo, exec_time);
    fprintf(fd, "%d\n", fast_algo);
  }

  for (int i = 0; i < 5 - gemm_num; i++) {
    fprintf(fd, "%d\n", (int)CUBLAS_GEMM_DEFAULT);
  }
  return 0;
}
